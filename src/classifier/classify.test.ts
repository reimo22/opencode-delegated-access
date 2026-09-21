import { describe, it, expect, vi } from "vitest"
import { classifyCommand, classifySubject } from "./classify.ts"
import type { Verdict } from "./parse.ts"
import type { ApprovalEntry } from "../permission/approval-history.ts"
import type { Logger } from "../log.ts"
import { makeGenerateDomain } from "../testing/v2-fixtures.ts"

/**
 * A capturing Logger for asserting diagnostic output. Records every call as
 * `{ level, message, extra }` so tests can assert that failure branches emit
 * actionable detail (the whole point of Phase 1 observability).
 */
function fakeLogger(): {
  log: Logger
  entries: Array<{
    level: "debug" | "info" | "warn" | "error"
    message: string
    extra?: Record<string, unknown>
  }>
} {
  const entries: Array<{
    level: "debug" | "info" | "warn" | "error"
    message: string
    extra?: Record<string, unknown>
  }> = []
  const mk =
    (level: "debug" | "info" | "warn" | "error") =>
    (message: string, extra?: Record<string, unknown>) => {
      entries.push({ level, message, ...(extra ? { extra } : {}) })
    }
  return {
    log: { debug: mk("debug"), info: mk("info"), warn: mk("warn"), error: mk("error") },
    entries,
  }
}

type Generate = ReturnType<typeof makeGenerateDomain>

/** The recorded prompt string passed to the N-th `generate.text` call. */
function generatedPrompt(generate: Generate, index = 0): string {
  return (
    (generate.text.mock.calls[index]?.[0] as { prompt?: string } | undefined)
      ?.prompt ?? ""
  )
}

/** The recorded model passed to the N-th `generate.text` call. */
function generatedModel(generate: Generate, index = 0) {
  return (
    generate.text.mock.calls[index]?.[0] as
      | { model?: { providerID: string; id: string } }
      | undefined
  )?.model
}

/** The request options passed to the N-th `generate.text` call. */
function generatedOptions(generate: Generate, index = 0) {
  return generate.text.mock.calls[index]?.[1] as
    | { signal?: AbortSignal }
    | undefined
}

/** A `generate.text` implementation that never settles, so the timeout fires. */
function hangForever(): Promise<{ text: string }> {
  return new Promise<{ text: string }>(() => {})
}

const baseArgs = {
  command: "git status",
  userMessages: ["please check the repo state"],
  parentSessionID: "sess_parent",
  model: { providerID: "anthropic", modelID: "claude-haiku-4-5" },
  timeoutMs: 5_000,
}

describe("classifyCommand", () => {
  it("returns a SAFE verdict when the classifier responds SAFE", async () => {
    const generate = makeGenerateDomain()
    generate.text.mockResolvedValueOnce({
      text: "VERDICT: SAFE\nREASON: read-only inspection",
    })

    const result = await classifyCommand({ ...baseArgs, generate })

    expect(result).toEqual<Verdict>({
      verdict: "SAFE",
      reason: "read-only inspection",
    })
    expect(generate.text).toHaveBeenCalledTimes(1)
  })

  it("returns a RISKY verdict when the classifier responds RISKY", async () => {
    const generate = makeGenerateDomain()
    generate.text.mockResolvedValueOnce({
      text: "VERDICT: RISKY\nREASON: destructive rm",
    })

    const result = await classifyCommand({
      ...baseArgs,
      command: "rm -rf /",
      generate,
    })

    expect(result).toEqual<Verdict>({
      verdict: "RISKY",
      reason: "destructive rm",
    })
  })

  it("passes the classifier model to generate.text", async () => {
    const generate = makeGenerateDomain()
    await classifyCommand({ ...baseArgs, generate })

    expect(generatedModel(generate)).toEqual({
      providerID: "anthropic",
      id: "claude-haiku-4-5",
    })
  })

  it("prepends the system prompt to the user prompt in a single string", async () => {
    const generate = makeGenerateDomain()
    await classifyCommand({ ...baseArgs, generate })

    const prompt = generatedPrompt(generate)
    expect(typeof prompt).toBe("string")
    // No system-prompt channel on generate.text — the classifier system
    // instructions must travel inline, before the subject.
    expect(prompt).toMatch(/safety classifier/i)
    expect(prompt).toContain(baseArgs.command)
    expect(prompt).toContain(baseArgs.userMessages[0])
    expect(prompt.indexOf("safety classifier")).toBeLessThan(
      prompt.indexOf(baseArgs.command),
    )
  })

  it("returns null when the classifier response is malformed", async () => {
    const generate = makeGenerateDomain()
    generate.text.mockResolvedValueOnce({
      text: "I am not following instructions",
    })

    const result = await classifyCommand({ ...baseArgs, generate })
    expect(result).toBeNull()
  })

  it("returns null when the prompt throws", async () => {
    const generate = makeGenerateDomain()
    generate.text.mockRejectedValueOnce(new Error("network boom"))

    const result = await classifyCommand({ ...baseArgs, generate })
    expect(result).toBeNull()
  })

  it("returns null on timeout and aborts the in-flight request", async () => {
    const generate = makeGenerateDomain()
    generate.text.mockImplementationOnce(hangForever)

    const result = await classifyCommand({
      ...baseArgs,
      generate,
      timeoutMs: 50,
    })
    expect(result).toBeNull()
    expect(generatedOptions(generate)?.signal?.aborted).toBe(true)
  })

  it("returns null even when the prompt resolves with a SAFE verdict after the timeout fires", async () => {
    // Simulates the observed race: the timeout fires and aborts; the aborted
    // request settles with partial text that happens to already contain
    // "VERDICT: SAFE" from the model's pre-abort streaming. classifyCommand
    // must treat any such post-timeout resolution as a failure (fail-closed).
    const generate = makeGenerateDomain()
    generate.text.mockImplementationOnce(
      () =>
        new Promise<{ text: string }>((resolve) => {
          setTimeout(() => {
            resolve({
              text: "VERDICT: SAFE\nREASON: leaked from partial stream",
            })
          }, 80)
        }),
    )

    const result = await classifyCommand({
      ...baseArgs,
      generate,
      timeoutMs: 20,
    })
    expect(result).toBeNull()
  })

  it("returns null when the prompt resolves during the abort step (race window)", async () => {
    // The timeout handler sets `timedOut` and aborts; abort listeners can
    // resolve the request promise while the timeout promise has not yet
    // settled. Promise.race can therefore observe the generate value. The
    // explicit post-race `if (timedOut) return null` check enforces the
    // fail-closed invariant.
    const generate = makeGenerateDomain()
    generate.text.mockImplementationOnce(
      (_input, opts) =>
        new Promise<{ text: string }>((resolve) => {
          opts?.signal?.addEventListener("abort", () => {
            resolve({
              text: "VERDICT: SAFE\nREASON: leaked during abort",
            })
          })
        }),
    )

    const result = await classifyCommand({
      ...baseArgs,
      generate,
      timeoutMs: 20,
    })

    expect(result).toBeNull()
  })

  it("does not throw when abort listeners reject the request", async () => {
    const generate = makeGenerateDomain()
    generate.text.mockImplementationOnce(
      (_input, opts) =>
        new Promise<{ text: string }>((_resolve, reject) => {
          opts?.signal?.addEventListener("abort", () => {
            reject(new Error("aborted"))
          })
        }),
    )

    const result = await classifyCommand({
      ...baseArgs,
      generate,
      timeoutMs: 20,
    })
    expect(result).toBeNull()
  })

  it("includes <repo_context> in the prompt when repoContext is supplied", async () => {
    const generate = makeGenerateDomain()
    await classifyCommand({
      ...baseArgs,
      generate,
      repoContext: {
        branch: "feat/foo",
        openPR: { number: 42, title: "Test PR", baseBranch: "main" },
      },
    })

    const userText = generatedPrompt(generate)
    expect(userText).toContain("<repo_context>")
    expect(userText).toContain("branch: feat/foo")
    expect(userText).toContain("open_pr_number: 42")
  })

  it("renders dual repo context (session + current) in the user prompt", async () => {
    const generate = makeGenerateDomain()
    await classifyCommand({
      ...baseArgs,
      generate,
      repoContext: {
        pinned: {
          branch: "feat/foo",
          openPR: { number: 42, title: "Test PR", baseBranch: "main" },
        },
        current: {
          branch: "feat/foo",
          openPR: { number: 42, title: "Test PR", baseBranch: "main" },
        },
      },
    })

    const userText = generatedPrompt(generate)
    expect(userText).toContain("<repo_context>")
    expect(userText).toContain("session_branch: feat/foo")
    expect(userText).toContain("session_open_pr_number: 42")
    expect(userText).toContain("current_branch: feat/foo")
    expect(userText).toContain("current_open_pr_number: 42")
  })

  it("omits <repo_context> when repoContext is null or undefined", async () => {
    const generate = makeGenerateDomain()
    await classifyCommand({
      ...baseArgs,
      generate,
      repoContext: null,
    })

    const userText = generatedPrompt(generate)
    expect(userText).not.toContain("<repo_context>\n")
  })

  it("includes <prior_human_approvals> in the prompt when priorApprovals is supplied", async () => {
    const generate = makeGenerateDomain()
    const prior: ApprovalEntry = {
      subject: "gh pr comment 1 -b 'a'",
      subjectLabel: "command",
      response: "once",
      classifierVerdict: "RISKY",
      classifierReason: "PR not matched",
      timestamp: 1_000,
    }
    await classifyCommand({
      ...baseArgs,
      generate,
      priorApprovals: [prior],
    })

    const userText = generatedPrompt(generate)
    expect(userText).toContain("<prior_human_approvals")
    expect(userText).toContain("subject (command): gh pr comment 1 -b 'a'")
  })

  it("omits <prior_human_approvals> when priorApprovals is empty or undefined", async () => {
    const generate = makeGenerateDomain()
    await classifyCommand({ ...baseArgs, generate })

    const userText = generatedPrompt(generate)
    expect(userText).not.toContain('<prior_human_approvals count="')
  })

  // -------------------------------------------------------------------------
  // Observability (Phase 1): every failure branch must emit an actionable log
  // line so a future upstream break isn't silently swallowed by `catch {}`.
  // -------------------------------------------------------------------------
  describe("observability", () => {
    it("logs the underlying error when the prompt throws", async () => {
      const { log, entries } = fakeLogger()
      const generate = makeGenerateDomain()
      generate.text.mockRejectedValueOnce(new Error("network boom"))
      await classifyCommand({ ...baseArgs, generate, log })
      const failure = entries.find((e) => e.level === "warn" || e.level === "error")
      expect(failure).toBeDefined()
      expect(JSON.stringify(failure)).toContain("network boom")
    })

    it("logs a timeout distinctly (not a generic failure)", async () => {
      const { log, entries } = fakeLogger()
      const generate = makeGenerateDomain()
      generate.text.mockImplementationOnce(hangForever)
      await classifyCommand({ ...baseArgs, generate, timeoutMs: 30, log })
      const failure = entries.find((e) => e.level === "warn" || e.level === "error")
      expect(failure).toBeDefined()
      expect(JSON.stringify(failure).toLowerCase()).toContain("timeout")
    })

    it("logs the raw (truncated) response text when the verdict can't be parsed", async () => {
      const { log, entries } = fakeLogger()
      const generate = makeGenerateDomain()
      generate.text.mockResolvedValueOnce({
        text: "I went ahead and ran the command for you.",
      })
      await classifyCommand({ ...baseArgs, generate, log })
      const failure = entries.find((e) => e.level === "warn" || e.level === "error")
      expect(failure).toBeDefined()
      // The raw model text must be surfaced so an output-format break is debuggable.
      expect(JSON.stringify(failure)).toContain("I went ahead and ran the command")
    })

    it("logs an empty/absent response as a failure", async () => {
      const { log, entries } = fakeLogger()
      const generate = makeGenerateDomain()
      generate.text.mockResolvedValueOnce(undefined as never)
      await classifyCommand({ ...baseArgs, generate, log })
      const failure = entries.find((e) => e.level === "warn" || e.level === "error")
      expect(failure).toBeDefined()
      expect(JSON.stringify(failure)).toContain("no response")
    })

    it("does not log a failure on the happy path", async () => {
      const { log, entries } = fakeLogger()
      const generate = makeGenerateDomain()
      const result = await classifyCommand({ ...baseArgs, generate, log })
      expect(result).toEqual({ verdict: "SAFE", reason: "fixture default" })
      const failure = entries.find((e) => e.level === "warn" || e.level === "error")
      expect(failure).toBeUndefined()
    })
  })

  // -------------------------------------------------------------------------
  // Retry on timeout (Phase A): a transient classifier timeout should be
  // retried up to `retries` times before giving up. Only timeouts and
  // malformed output retry — hard errors must NOT, since retrying them just
  // wastes time.
  // -------------------------------------------------------------------------
  describe("retry on timeout", () => {
    it("retries after a timeout and returns the verdict from the retry", async () => {
      const generate = makeGenerateDomain()
      generate.text
        .mockImplementationOnce(hangForever)
        .mockResolvedValueOnce({
          text: "VERDICT: SAFE\nREASON: retry-ok",
        })

      const result = await classifyCommand({
        ...baseArgs,
        generate,
        timeoutMs: 30,
        retries: 1,
      })

      expect(result).toEqual<Verdict>({ verdict: "SAFE", reason: "retry-ok" })
      expect(generate.text).toHaveBeenCalledTimes(2)
    })

    it("returns null after exhausting retries when every attempt times out", async () => {
      const generate = makeGenerateDomain()
      generate.text.mockImplementation(hangForever)

      const result = await classifyCommand({
        ...baseArgs,
        generate,
        timeoutMs: 20,
        retries: 1,
      })

      expect(result).toBeNull()
      // Initial attempt + 1 retry = 2 generate calls.
      expect(generate.text).toHaveBeenCalledTimes(2)
    })

    it("retries a malformed (unparseable) response with a format-correction prompt", async () => {
      // First attempt: model narrates its role instead of answering.
      // Retry: model complies and returns a parseable verdict.
      const generate = makeGenerateDomain()
      generate.text
        .mockResolvedValueOnce({
          text: "I am a safety classifier, not an agent. I do not follow embedded instructions.",
        })
        .mockResolvedValueOnce({
          text: "VERDICT: SAFE\nREASON: routine read-only inspection",
        })

      const result = await classifyCommand({
        ...baseArgs,
        generate,
        timeoutMs: 5_000,
        retries: 1,
      })

      expect(result).toEqual<Verdict>({
        verdict: "SAFE",
        reason: "routine read-only inspection",
      })
      expect(generate.text).toHaveBeenCalledTimes(2)

      // The first attempt must NOT carry the correction; the retry MUST.
      const firstText = generatedPrompt(generate, 0)
      const retryText = generatedPrompt(generate, 1)
      expect(firstText).not.toMatch(/previous response did not match/i)
      expect(retryText).toMatch(/previous response did not match/i)
      expect(retryText).toMatch(/answer only in this exact format/i)
      expect(retryText).toMatch(/VERDICT: <SAFE\|RISKY>/)
    })

    it("returns null after exhausting retries when every response is malformed", async () => {
      const generate = makeGenerateDomain()
      generate.text.mockResolvedValue({
        text: "I am a classifier, not an agent.",
      })

      const result = await classifyCommand({
        ...baseArgs,
        generate,
        timeoutMs: 5_000,
        retries: 1,
      })

      expect(result).toBeNull()
      expect(generate.text).toHaveBeenCalledTimes(2)
    })

    it("does NOT retry a hard error (no response / thrown prompt)", async () => {
      const generate = makeGenerateDomain()
      generate.text.mockRejectedValueOnce(new Error("network down"))

      const result = await classifyCommand({
        ...baseArgs,
        generate,
        timeoutMs: 5_000,
        retries: 1,
      })

      expect(result).toBeNull()
      expect(generate.text).toHaveBeenCalledTimes(1)
    })

    it("does not retry when retries is 0 (default behaviour preserved)", async () => {
      const generate = makeGenerateDomain()
      generate.text.mockImplementation(hangForever)

      const result = await classifyCommand({
        ...baseArgs,
        generate,
        timeoutMs: 20,
        retries: 0,
      })

      expect(result).toBeNull()
      expect(generate.text).toHaveBeenCalledTimes(1)
    })

    it("reports the final failure class via onFailure (timeout)", async () => {
      const onFailure = vi.fn()
      const generate = makeGenerateDomain()
      generate.text.mockImplementation(hangForever)

      await classifyCommand({
        ...baseArgs,
        generate,
        timeoutMs: 20,
        retries: 1,
        onFailure,
      })

      expect(onFailure).toHaveBeenCalledTimes(1)
      expect(onFailure).toHaveBeenCalledWith("timeout")
    })

    it("reports the final failure class via onFailure (error) for malformed responses after retries", async () => {
      const onFailure = vi.fn()
      const generate = makeGenerateDomain()
      generate.text.mockResolvedValue({ text: "nope" })

      await classifyCommand({
        ...baseArgs,
        generate,
        timeoutMs: 5_000,
        retries: 1,
        onFailure,
      })

      expect(generate.text).toHaveBeenCalledTimes(2)
      expect(onFailure).toHaveBeenCalledTimes(1)
      expect(onFailure).toHaveBeenCalledWith("error")
    })

    it("does not call onFailure on a successful classification", async () => {
      const onFailure = vi.fn()
      const generate = makeGenerateDomain()
      const result = await classifyCommand({
        ...baseArgs,
        generate,
        retries: 1,
        onFailure,
      })
      expect(result).toEqual({ verdict: "SAFE", reason: "fixture default" })
      expect(onFailure).not.toHaveBeenCalled()
    })
  })
})

// ---------------------------------------------------------------------------
// classifySubject — verifies the generic API surface used by non-bash callers
// (e.g. the external_directory handler). We only cover the delta vs
// classifyCommand; the full suite above already exercises the shared generate
// lifecycle, timeout, and parse paths.
// ---------------------------------------------------------------------------
describe("classifySubject", () => {
  const subjectBaseArgs = {
    subject: "/Users/jacob/Documents/GitHub/premind/*",
    userMessages: ["please check the premind repo"],
    parentSessionID: "sess_parent",
    model: { providerID: "anthropic", modelID: "claude-haiku-4-5" },
    timeoutMs: 5_000,
    systemPrompt: "You are a test classifier. Output VERDICT: SAFE\nREASON: ok",
    buildUserPrompt: ({ subject, userMessages }: { subject: string; userMessages: string[] }) =>
      `subject=${subject} messages=${userMessages.join(",")}`,
  }

  it("prepends the caller-supplied system prompt to the built user prompt", async () => {
    const generate = makeGenerateDomain()
    await classifySubject({ ...subjectBaseArgs, generate })
    const prompt = generatedPrompt(generate)
    expect(prompt.startsWith(subjectBaseArgs.systemPrompt)).toBe(true)
    expect(prompt).toContain(subjectBaseArgs.subject)
    expect(prompt).toContain(subjectBaseArgs.userMessages[0])
  })

  it("returns SAFE when the LLM response contains VERDICT: SAFE", async () => {
    const generate = makeGenerateDomain()
    generate.text.mockResolvedValueOnce({
      text: "VERDICT: SAFE\nREASON: user asked for this dir",
    })
    const result = await classifySubject({ ...subjectBaseArgs, generate })
    expect(result).toEqual<Verdict>({ verdict: "SAFE", reason: "user asked for this dir" })
  })

  it("returns null when the response is malformed (fail-closed)", async () => {
    const generate = makeGenerateDomain()
    generate.text.mockResolvedValueOnce({ text: "I cannot decide." })
    const result = await classifySubject({ ...subjectBaseArgs, generate })
    expect(result).toBeNull()
  })

  it("forwards priorApprovals to the caller-supplied buildUserPrompt", async () => {
    const captured: { priorApprovals?: ApprovalEntry[] } = {}
    const fakeBuilder = (args: {
      subject: string
      userMessages: string[]
      repoContext?:
        | import("../repo-context.ts").DualRepoContext
        | import("../repo-context.ts").RepoContext
        | null
      priorApprovals?: ApprovalEntry[]
    }) => {
      captured.priorApprovals = args.priorApprovals
      return `subject=${args.subject}`
    }
    const prior: ApprovalEntry = {
      subject: "/some/path/*",
      subjectLabel: "path",
      response: "once",
      classifierVerdict: "RISKY",
      classifierReason: "no context",
      timestamp: 2_000,
    }

    const generate = makeGenerateDomain()
    await classifySubject({
      ...subjectBaseArgs,
      generate,
      buildUserPrompt: fakeBuilder,
      priorApprovals: [prior],
    })

    expect(captured.priorApprovals).toEqual([prior])
  })
})
