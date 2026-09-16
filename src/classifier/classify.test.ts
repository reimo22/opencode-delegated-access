import { describe, it, expect, vi } from "vitest"
import { classifyCommand, classifySubject } from "./classify.ts"
import type { Verdict } from "./parse.ts"
import type { ApprovalEntry } from "../permission/approval-history.ts"
import type { Logger } from "../log.ts"
import { makeSessionDomain } from "../testing/v2-fixtures.ts"

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

type Session = ReturnType<typeof makeSessionDomain>

/** The recorded input to the N-th `session.create` call. */
function createdWith(session: Session, index = 0) {
  return session.create.mock.calls[index]?.[0] as
    | { title?: string; model?: { providerID: string; id: string } }
    | undefined
}

/** The recorded user-prompt string passed to the N-th `session.generate` call. */
function generatedPrompt(session: Session, index = 0): string {
  return (
    (session.generate.mock.calls[index]?.[0] as { prompt?: string } | undefined)
      ?.prompt ?? ""
  )
}

/**
 * Resolve the next `session.create` call with no session id. The fixture's
 * `create` mock is typed to the happy-path `{ id: string }` shape, so the cast
 * is needed to exercise the id-less branch the V2 domain can return.
 */
function createWithoutID(session: Session) {
  session.create.mockResolvedValueOnce(undefined as never)
}

/** A `session.generate` implementation that never settles, so the timeout fires. */
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
    const session = makeSessionDomain()
    session.generate.mockResolvedValueOnce({
      text: "VERDICT: SAFE\nREASON: read-only inspection",
    })

    const result = await classifyCommand({ ...baseArgs, session })

    expect(result).toEqual<Verdict>({
      verdict: "SAFE",
      reason: "read-only inspection",
    })
    expect(session.create).toHaveBeenCalledTimes(1)
    expect(session.generate).toHaveBeenCalledTimes(1)
  })

  it("returns a RISKY verdict when the classifier responds RISKY", async () => {
    const session = makeSessionDomain()
    session.generate.mockResolvedValueOnce({
      text: "VERDICT: RISKY\nREASON: destructive rm",
    })

    const result = await classifyCommand({
      ...baseArgs,
      command: "rm -rf /",
      session,
    })

    expect(result).toEqual<Verdict>({
      verdict: "RISKY",
      reason: "destructive rm",
    })
  })

  it("passes the classifier model and a labelled title to session.create", async () => {
    const session = makeSessionDomain()
    await classifyCommand({ ...baseArgs, session })

    const arg = createdWith(session)
    // V2 create has no parentID — the session is top-level and identified by
    // its title. The classifier model is mapped from ModelRef to the domain's
    // `{ providerID, id }` shape.
    expect(arg?.model).toEqual({
      providerID: "anthropic",
      id: "claude-haiku-4-5",
    })
    expect(arg?.title).toMatch(/delegated-access|classifier/i)
  })

  it("sends the built user prompt as a single string to session.generate", async () => {
    const session = makeSessionDomain()
    await classifyCommand({ ...baseArgs, session })

    const prompt = generatedPrompt(session)
    expect(typeof prompt).toBe("string")
    expect(prompt).toContain(baseArgs.command)
    expect(prompt).toContain(baseArgs.userMessages[0])
    // The generate call targets the ephemeral session the classifier created.
    const generateArg = session.generate.mock.calls[0]?.[0] as
      | { sessionID?: string }
      | undefined
    expect(generateArg?.sessionID).toBe("sess_ephemeral")
  })

  it("returns null when the classifier response is malformed", async () => {
    const session = makeSessionDomain()
    session.generate.mockResolvedValueOnce({
      text: "I am not following instructions",
    })

    const result = await classifyCommand({ ...baseArgs, session })
    expect(result).toBeNull()
  })

  it("returns null when session.create throws (and does not generate)", async () => {
    const session = makeSessionDomain()
    session.create.mockRejectedValueOnce(new Error("cannot create"))

    const result = await classifyCommand({ ...baseArgs, session })
    expect(result).toBeNull()
    expect(session.generate).not.toHaveBeenCalled()
  })

  it("returns null when session.create returns no session id", async () => {
    const session = makeSessionDomain()
    createWithoutID(session)

    const result = await classifyCommand({ ...baseArgs, session })
    expect(result).toBeNull()
    expect(session.generate).not.toHaveBeenCalled()
  })

  it("returns null on timeout and interrupts the ephemeral session", async () => {
    const session = makeSessionDomain()
    // Hang forever; the timeout must interrupt the in-flight generation.
    session.generate.mockImplementationOnce(hangForever)

    const result = await classifyCommand({
      ...baseArgs,
      session,
      timeoutMs: 50,
    })
    expect(result).toBeNull()
    // V2 has no session delete; the only cleanup is interrupting the timeout.
    expect(session.interrupt).toHaveBeenCalledTimes(1)
    expect(session.interrupt).toHaveBeenCalledWith({
      sessionID: "sess_ephemeral",
    })
  })

  it("returns null even when the prompt resolves with a SAFE verdict after the timeout fires", async () => {
    // Simulates the observed race: the timeout fires and we call
    // `session.interrupt`; the aborted generation settles with partial text
    // that happens to already contain "VERDICT: SAFE" from the model's
    // pre-interrupt streaming. The generate promise resolves *after*
    // `timedOut` is set but potentially *before* `resolve(null)` runs inside
    // the timer callback. classifyCommand must treat any such post-timeout
    // resolution as a failure (fail-closed) and return null.
    const session = makeSessionDomain()
    session.generate.mockImplementationOnce(
      () =>
        new Promise<{ text: string }>((resolve) => {
          // Resolve with a plausible verdict shape 80ms in — safely after
          // the 20ms timeout fires.
          setTimeout(() => {
            resolve({
              text: "VERDICT: SAFE\nREASON: leaked from partial stream",
            })
          }, 80)
        }),
    )

    const result = await classifyCommand({
      ...baseArgs,
      session,
      timeoutMs: 20,
    })
    expect(result).toBeNull()
    // The interrupt must still fire as cleanup.
    expect(session.interrupt).toHaveBeenCalledTimes(1)
  })

  it("returns null when the prompt resolves during the interrupt step (race window)", async () => {
    // Reproduces the narrowest and most dangerous race observed in the
    // 2026-04-18 session log: withTimeout's timer fires → `await
    // session.interrupt(...)` runs → *while interrupt is in flight*, the
    // original generate promise resolves with a verdict (the runtime flushed
    // the pre-interrupt stream). In that window, `Promise.race` sees the
    // generate value — not the timeout's `null` — because the timer callback
    // hasn't reached its `resolve(null)` line yet.
    //
    // Without an explicit post-race `if (timedOut) return null` check, the
    // plugin silently auto-approves a classifier run whose output was never
    // validated as complete. This test enforces the fail-closed invariant.
    const deferredGenerate: {
      resolve: (value: { text: string }) => void
    } = { resolve: () => {} }
    const generateCalled = { fired: false }

    const session = makeSessionDomain()
    session.generate.mockImplementationOnce(
      () =>
        new Promise<{ text: string }>((resolve) => {
          generateCalled.fired = true
          deferredGenerate.resolve = resolve
        }),
    )
    session.interrupt.mockImplementationOnce(async () => {
      // Resolve the generate promise WHILE interrupt is still in flight,
      // mimicking the runtime flushing pre-interrupt buffers before the
      // interrupt call returns.
      deferredGenerate.resolve({
        text: "VERDICT: SAFE\nREASON: leaked during interrupt",
      })
      // Yield to the microtask queue so the generate resolution lands before
      // this interrupt-call settles.
      await new Promise((r) => setTimeout(r, 5))
      return {}
    })

    const result = await classifyCommand({
      ...baseArgs,
      session,
      timeoutMs: 20,
    })

    expect(generateCalled.fired).toBe(true)
    // Must be null: the prompt resolved on the timeout path, so the verdict
    // is untrustworthy even though its text parses cleanly.
    expect(result).toBeNull()
    expect(session.interrupt).toHaveBeenCalledTimes(1)
  })

  it("swallows interrupt errors on the timeout path (best-effort cleanup must not mask the outcome)", async () => {
    const session = makeSessionDomain()
    session.generate.mockImplementationOnce(hangForever)
    session.interrupt.mockRejectedValueOnce(new Error("interrupt failed"))

    const result = await classifyCommand({
      ...baseArgs,
      session,
      timeoutMs: 20,
    })
    // The interrupt rejection is swallowed; the fail-closed timeout stands.
    expect(result).toBeNull()
    expect(session.interrupt).toHaveBeenCalledTimes(1)
  })

  it("includes <repo_context> in the prompt when repoContext is supplied", async () => {
    const session = makeSessionDomain()
    await classifyCommand({
      ...baseArgs,
      session,
      repoContext: {
        branch: "feat/foo",
        openPR: { number: 42, title: "Test PR", baseBranch: "main" },
      },
    })

    const userText = generatedPrompt(session)
    expect(userText).toContain("<repo_context>")
    expect(userText).toContain("branch: feat/foo")
    expect(userText).toContain("open_pr_number: 42")
  })

  it("renders dual repo context (session + current) in the user prompt", async () => {
    const session = makeSessionDomain()
    await classifyCommand({
      ...baseArgs,
      session,
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

    const userText = generatedPrompt(session)
    expect(userText).toContain("<repo_context>")
    expect(userText).toContain("session_branch: feat/foo")
    expect(userText).toContain("session_open_pr_number: 42")
    expect(userText).toContain("current_branch: feat/foo")
    expect(userText).toContain("current_open_pr_number: 42")
  })

  it("omits <repo_context> when repoContext is null or undefined", async () => {
    const session = makeSessionDomain()
    await classifyCommand({
      ...baseArgs,
      session,
      repoContext: null,
    })

    const userText = generatedPrompt(session)
    expect(userText).not.toContain("<repo_context>")
  })

  it("includes <prior_human_approvals> in the prompt when priorApprovals is supplied", async () => {
    const session = makeSessionDomain()
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
      session,
      priorApprovals: [prior],
    })

    const userText = generatedPrompt(session)
    expect(userText).toContain("<prior_human_approvals")
    expect(userText).toContain("subject (command): gh pr comment 1 -b 'a'")
  })

  it("omits <prior_human_approvals> when priorApprovals is empty or undefined", async () => {
    const session = makeSessionDomain()
    await classifyCommand({ ...baseArgs, session })

    const userText = generatedPrompt(session)
    expect(userText).not.toContain("<prior_human_approvals")
  })

  it("invokes onEphemeralSessionCreated and onEphemeralSessionDeleted around the classifier call", async () => {
    const session = makeSessionDomain()
    const created = vi.fn()
    const deleted = vi.fn()

    await classifyCommand({
      ...baseArgs,
      session,
      onEphemeralSessionCreated: created,
      onEphemeralSessionDeleted: deleted,
    })

    expect(created).toHaveBeenCalledTimes(1)
    // created now receives (id, systemPrompt); assert the id positionally.
    expect(created.mock.calls[0]?.[0]).toBe("sess_ephemeral")
    expect(deleted).toHaveBeenCalledTimes(1)
    expect(deleted).toHaveBeenCalledWith("sess_ephemeral")
    // Order: created before deleted.
    const createdOrder = created.mock.invocationCallOrder[0] ?? 0
    const deletedOrder = deleted.mock.invocationCallOrder[0] ?? 0
    expect(createdOrder).toBeLessThan(deletedOrder)
  })

  it("still invokes onEphemeralSessionDeleted when the prompt throws", async () => {
    const session = makeSessionDomain()
    session.generate.mockRejectedValueOnce(new Error("boom"))
    const created = vi.fn()
    const deleted = vi.fn()

    await classifyCommand({
      ...baseArgs,
      session,
      onEphemeralSessionCreated: created,
      onEphemeralSessionDeleted: deleted,
    })

    expect(created).toHaveBeenCalledTimes(1)
    expect(deleted).toHaveBeenCalledTimes(1)
  })

  it("does NOT invoke onEphemeralSessionCreated when session.create fails", async () => {
    const session = makeSessionDomain()
    session.create.mockRejectedValueOnce(new Error("cannot create"))
    const created = vi.fn()
    const deleted = vi.fn()

    await classifyCommand({
      ...baseArgs,
      session,
      onEphemeralSessionCreated: created,
      onEphemeralSessionDeleted: deleted,
    })

    expect(created).not.toHaveBeenCalled()
    expect(deleted).not.toHaveBeenCalled()
  })

  it("passes the system prompt alongside the session id to onEphemeralSessionCreated", async () => {
    const session = makeSessionDomain()
    const created = vi.fn()

    await classifyCommand({
      ...baseArgs,
      session,
      onEphemeralSessionCreated: created,
    })

    expect(created).toHaveBeenCalledTimes(1)
    const [id, systemPrompt] = created.mock.calls[0] ?? []
    expect(id).toBe("sess_ephemeral")
    // The classifier's bash system prompt must be supplied so the caller can
    // register it for the system-transform isolation hook.
    expect(typeof systemPrompt).toBe("string")
    expect(systemPrompt).toMatch(/safety classifier/i)
  })

  // -------------------------------------------------------------------------
  // Observability (Phase 1): every failure branch must emit an actionable log
  // line so a future upstream break isn't silently swallowed by `catch {}`.
  // -------------------------------------------------------------------------
  describe("observability", () => {
    it("logs the underlying error when session.create throws", async () => {
      const { log, entries } = fakeLogger()
      const session = makeSessionDomain()
      session.create.mockRejectedValueOnce(new Error("cannot create"))
      await classifyCommand({ ...baseArgs, session, log })
      const failure = entries.find((e) => e.level === "warn" || e.level === "error")
      expect(failure).toBeDefined()
      expect(JSON.stringify(failure)).toContain("cannot create")
    })

    it("logs when session.create returns no session id", async () => {
      const { log, entries } = fakeLogger()
      const session = makeSessionDomain()
      createWithoutID(session)
      await classifyCommand({ ...baseArgs, session, log })
      const failure = entries.find((e) => e.level === "warn" || e.level === "error")
      expect(failure).toBeDefined()
      expect(failure?.message.toLowerCase()).toContain("session")
    })

    it("logs the underlying error when the prompt throws", async () => {
      const { log, entries } = fakeLogger()
      const session = makeSessionDomain()
      session.generate.mockRejectedValueOnce(new Error("network boom"))
      await classifyCommand({ ...baseArgs, session, log })
      const failure = entries.find((e) => e.level === "warn" || e.level === "error")
      expect(failure).toBeDefined()
      expect(JSON.stringify(failure)).toContain("network boom")
    })

    it("logs a timeout distinctly (not a generic failure)", async () => {
      const { log, entries } = fakeLogger()
      const session = makeSessionDomain()
      session.generate.mockImplementationOnce(hangForever)
      await classifyCommand({ ...baseArgs, session, timeoutMs: 30, log })
      const failure = entries.find((e) => e.level === "warn" || e.level === "error")
      expect(failure).toBeDefined()
      expect(JSON.stringify(failure).toLowerCase()).toContain("timeout")
    })

    it("logs the raw (truncated) response text when the verdict can't be parsed", async () => {
      const { log, entries } = fakeLogger()
      const session = makeSessionDomain()
      session.generate.mockResolvedValueOnce({
        text: "I went ahead and ran the command for you.",
      })
      await classifyCommand({ ...baseArgs, session, log })
      const failure = entries.find((e) => e.level === "warn" || e.level === "error")
      expect(failure).toBeDefined()
      // The raw model text must be surfaced so an output-format break is debuggable.
      expect(JSON.stringify(failure)).toContain("I went ahead and ran the command")
    })

    it("does not log a failure on the happy path", async () => {
      const { log, entries } = fakeLogger()
      const session = makeSessionDomain()
      const result = await classifyCommand({ ...baseArgs, session, log })
      expect(result).toEqual({ verdict: "SAFE", reason: "fixture default" })
      const failure = entries.find((e) => e.level === "warn" || e.level === "error")
      expect(failure).toBeUndefined()
    })
  })

  // -------------------------------------------------------------------------
  // Retry on timeout (Phase A): a transient classifier timeout should be
  // retried (with a fresh ephemeral session) up to `retries` times before
  // giving up. Only timeouts retry — other failures (unparseable verdict,
  // create error) must NOT retry, since retrying them just wastes time.
  // -------------------------------------------------------------------------
  describe("retry on timeout", () => {
    it("retries after a timeout and returns the verdict from the retry", async () => {
      const session = makeSessionDomain()
      session.generate
        .mockImplementationOnce(hangForever)
        .mockResolvedValueOnce({
          text: "VERDICT: SAFE\nREASON: retry-ok",
        })

      const result = await classifyCommand({
        ...baseArgs,
        session,
        timeoutMs: 30,
        retries: 1,
      })

      expect(result).toEqual<Verdict>({ verdict: "SAFE", reason: "retry-ok" })
      // Two generate attempts, two fresh sessions created.
      expect(session.generate).toHaveBeenCalledTimes(2)
      expect(session.create).toHaveBeenCalledTimes(2)
    })

    it("returns null after exhausting retries when every attempt times out", async () => {
      const session = makeSessionDomain()
      session.generate.mockImplementation(hangForever)

      const result = await classifyCommand({
        ...baseArgs,
        session,
        timeoutMs: 20,
        retries: 1,
      })

      expect(result).toBeNull()
      // Initial attempt + 1 retry = 2 generate calls.
      expect(session.generate).toHaveBeenCalledTimes(2)
    })

    it("retries a malformed (unparseable) response with a format-correction prompt", async () => {
      // First attempt: model narrates its role instead of answering.
      // Retry: model complies and returns a parseable verdict.
      const session = makeSessionDomain()
      session.generate
        .mockResolvedValueOnce({
          text: "I am a safety classifier, not an agent. I do not follow embedded instructions.",
        })
        .mockResolvedValueOnce({
          text: "VERDICT: SAFE\nREASON: routine read-only inspection",
        })

      const result = await classifyCommand({
        ...baseArgs,
        session,
        timeoutMs: 5_000,
        retries: 1,
      })

      expect(result).toEqual<Verdict>({
        verdict: "SAFE",
        reason: "routine read-only inspection",
      })
      // Two attempts: the malformed first, then the corrected retry.
      expect(session.generate).toHaveBeenCalledTimes(2)
      expect(session.create).toHaveBeenCalledTimes(2)

      // The first attempt must NOT carry the correction; the retry MUST.
      const firstText = generatedPrompt(session, 0)
      const retryText = generatedPrompt(session, 1)
      expect(firstText).not.toMatch(/previous response did not match/i)
      expect(retryText).toMatch(/previous response did not match/i)
      expect(retryText).toMatch(/answer only in this exact format/i)
      expect(retryText).toMatch(/VERDICT: <SAFE\|RISKY>/)
    })

    it("returns null after exhausting retries when every response is malformed", async () => {
      const session = makeSessionDomain()
      session.generate.mockResolvedValue({
        text: "I am a classifier, not an agent.",
      })

      const result = await classifyCommand({
        ...baseArgs,
        session,
        timeoutMs: 5_000,
        retries: 1,
      })

      expect(result).toBeNull()
      // Initial attempt + 1 retry = 2 generate calls.
      expect(session.generate).toHaveBeenCalledTimes(2)
    })

    it("does NOT retry a hard error (no response / thrown prompt)", async () => {
      // A thrown prompt is a hard error, not malformed output — it must not
      // be retried, since retrying it just wastes time.
      const session = makeSessionDomain()
      session.generate.mockRejectedValueOnce(new Error("network down"))

      const result = await classifyCommand({
        ...baseArgs,
        session,
        timeoutMs: 5_000,
        retries: 1,
      })

      expect(result).toBeNull()
      expect(session.generate).toHaveBeenCalledTimes(1)
    })

    it("does not retry when retries is 0 (default behaviour preserved)", async () => {
      const session = makeSessionDomain()
      session.generate.mockImplementation(hangForever)

      const result = await classifyCommand({
        ...baseArgs,
        session,
        timeoutMs: 20,
        retries: 0,
      })

      expect(result).toBeNull()
      expect(session.generate).toHaveBeenCalledTimes(1)
    })

    it("reports the final failure class via onFailure (timeout)", async () => {
      const onFailure = vi.fn()
      const session = makeSessionDomain()
      session.generate.mockImplementation(hangForever)

      await classifyCommand({
        ...baseArgs,
        session,
        timeoutMs: 20,
        retries: 1,
        onFailure,
      })

      expect(onFailure).toHaveBeenCalledTimes(1)
      expect(onFailure).toHaveBeenCalledWith("timeout")
    })

    it("reports the final failure class via onFailure (error) for malformed responses after retries", async () => {
      // Malformed output is retried, but once retries are exhausted it is
      // reported to the caller as the "error" failure class (the public
      // ClassifyFailureClass surface stays timeout|error).
      const onFailure = vi.fn()
      const session = makeSessionDomain()
      session.generate.mockResolvedValue({ text: "nope" })

      await classifyCommand({
        ...baseArgs,
        session,
        timeoutMs: 5_000,
        retries: 1,
        onFailure,
      })

      // Retried once (2 attempts), then reported error exactly once.
      expect(session.generate).toHaveBeenCalledTimes(2)
      expect(onFailure).toHaveBeenCalledTimes(1)
      expect(onFailure).toHaveBeenCalledWith("error")
    })

    it("does not call onFailure on a successful classification", async () => {
      const onFailure = vi.fn()
      const session = makeSessionDomain()
      const result = await classifyCommand({
        ...baseArgs,
        session,
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
// classifyCommand; the full suite above already exercises the shared session
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

  it("hands the caller-supplied system prompt to onEphemeralSessionCreated", async () => {
    // V2 never sends `systemPrompt` on the generate request; it is handed to
    // the caller so the session.hook("context") handler can apply it.
    const session = makeSessionDomain()
    const created = vi.fn()
    await classifySubject({
      ...subjectBaseArgs,
      session,
      onEphemeralSessionCreated: created,
    })
    expect(created).toHaveBeenCalledTimes(1)
    expect(created.mock.calls[0]?.[1]).toBe(subjectBaseArgs.systemPrompt)
  })

  it("passes the buildUserPrompt output as the generate prompt", async () => {
    const session = makeSessionDomain()
    await classifySubject({ ...subjectBaseArgs, session })
    const prompt = generatedPrompt(session)
    expect(prompt).toContain(subjectBaseArgs.subject)
    expect(prompt).toContain(subjectBaseArgs.userMessages[0])
  })

  it("returns SAFE when the LLM response contains VERDICT: SAFE", async () => {
    const session = makeSessionDomain()
    session.generate.mockResolvedValueOnce({
      text: "VERDICT: SAFE\nREASON: user asked for this dir",
    })
    const result = await classifySubject({ ...subjectBaseArgs, session })
    expect(result).toEqual<Verdict>({ verdict: "SAFE", reason: "user asked for this dir" })
  })

  it("returns null when the response is malformed (fail-closed)", async () => {
    const session = makeSessionDomain()
    session.generate.mockResolvedValueOnce({ text: "I cannot decide." })
    const result = await classifySubject({ ...subjectBaseArgs, session })
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

    const session = makeSessionDomain()
    await classifySubject({
      ...subjectBaseArgs,
      session,
      buildUserPrompt: fakeBuilder,
      priorApprovals: [prior],
    })

    expect(captured.priorApprovals).toEqual([prior])
  })
})
