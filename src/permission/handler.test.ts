import { describe, it, expect, vi, beforeEach } from "vitest"

vi.mock("../classifier/classify.ts", () => ({
  classifyCommand: vi.fn(),
  // V2 routes external_directory through the directory classifier.
  classifyDirectory: vi.fn(),
}))
// Only `getSessionMessages` is mocked (it's the only I/O call the handler
// performs against the messages module). The pure extractors
// (extractLastUserMessages / extractLatestAssistantModel) run unmocked so
// tests exercise the same code path as production.
vi.mock("../ui/messages.ts", async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown>
  return {
    ...actual,
    getSessionMessages: vi.fn(),
  }
})
// `resolveRootSessionID` walks the session parent chain via the V2 session
// domain; mock it so handler tests control what "root" sessionID the handler
// sees without having to stub session.get.
vi.mock("../ui/session-tree.ts", () => ({
  resolveRootSessionID: vi.fn(),
}))
vi.mock("./safe-path.ts", () => ({
  runSafePath: vi.fn(),
}))
vi.mock("./risky-path.ts", () => ({
  runRiskyPathInBackground: vi.fn(),
}))
// Mock only the notification runner; keep the real FailureNotifyRateLimiter so
// the handler's rate-limit wiring is exercised end-to-end.
vi.mock("./failure-notify.ts", async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown>
  return {
    ...actual,
    runFailureNotificationInBackground: vi.fn(async () => {}),
  }
})

import { classifyCommand, classifyDirectory } from "../classifier/classify.ts"
import { getSessionMessages } from "../ui/messages.ts"
import { resolveRootSessionID } from "../ui/session-tree.ts"
import { runSafePath } from "./safe-path.ts"
import { runRiskyPathInBackground } from "./risky-path.ts"
import {
  runFailureNotificationInBackground,
  FailureNotifyRateLimiter,
} from "./failure-notify.ts"
import {
  handlePermissionEvent,
  evaluationKey,
  type HandlerContext,
  type PermissionEvaluation,
} from "./handler.ts"
import { DirectoryVerdictCache } from "./directory-cache.ts"
import { ApprovalHistoryStore } from "./approval-history.ts"
import { PendingSubjectsMap } from "./pending-subjects.ts"
import { parseConfig, type DelegatedAccessConfig } from "../config.ts"
import {
  makeHandlerContext,
  makeEvaluation,
  userMessage,
  assistantMessage,
  SAMPLE_MODEL,
} from "../testing/v2-fixtures.ts"

const mockedClassify = vi.mocked(classifyCommand)
const mockedClassifyDirectory = vi.mocked(classifyDirectory)
const mockedGetSessionMessages = vi.mocked(getSessionMessages)
const mockedResolveRoot = vi.mocked(resolveRootSessionID)
const mockedSafe = vi.mocked(runSafePath)
const mockedRisky = vi.mocked(runRiskyPathInBackground)
const mockedFailureNotify = vi.mocked(runFailureNotificationInBackground)

/** Options for the local ctx builder: HandlerContext overrides plus a partial
 * config, so tests can flip individual config flags without rebuilding the
 * whole config object. */
type BuildCtxOptions = Partial<Omit<HandlerContext, "config">> & {
  config?: Partial<DelegatedAccessConfig>
}

/**
 * Build a V2 HandlerContext on top of the shared fixtures. Defaults
 * `sessionModel` to a usable model so classification proceeds unless a test
 * explicitly passes `sessionModel: undefined`.
 */
function buildCtx(overrides: BuildCtxOptions = {}): HandlerContext {
  const { config, ...rest } = overrides
  return makeHandlerContext({
    sessionModel: SAMPLE_MODEL,
    ...rest,
    ...(config ? { config: { ...parseConfig(undefined), ...config } } : {}),
  })
}

beforeEach(() => {
  mockedClassify.mockReset()
  mockedClassifyDirectory.mockReset()
  mockedGetSessionMessages.mockReset()
  mockedResolveRoot.mockReset()
  mockedSafe.mockReset()
  mockedRisky.mockReset()
  mockedFailureNotify.mockReset()
  mockedFailureNotify.mockResolvedValue(undefined)
  // Default: one user message, no assistant messages. Tests that need
  // assistant-model fallback override this with their own value.
  mockedGetSessionMessages.mockResolvedValue([userMessage("please check the repo")])
  // Default: treat the permission's own sessionID as the root (i.e. not
  // a subagent). Subagent tests override this to return a different
  // sessionID. Fail-closed tests override it to return null.
  mockedResolveRoot.mockImplementation(async (_session, sessionID) => sessionID)
})

describe("handlePermissionEvent", () => {
  it("does nothing when config.enabled is false", async () => {
    const ctx = buildCtx({ config: { enabled: false } })
    const ev = makeEvaluation()
    await handlePermissionEvent(ev, ctx)
    expect(mockedClassify).not.toHaveBeenCalled()
    expect(ev.effect).toBe("ask")
  })

  it("does nothing for non-shell permission actions", async () => {
    const ctx = buildCtx()
    const ev = makeEvaluation({ action: "edit" })
    await handlePermissionEvent(ev, ctx)
    expect(mockedClassify).not.toHaveBeenCalled()
    expect(ev.effect).toBe("ask")
  })

  it("sets ev.effect='allow' when SAFE and safe-path returns allow (no SDK reply)", async () => {
    mockedClassify.mockResolvedValueOnce({
      verdict: "SAFE",
      reason: "read-only",
    })
    mockedSafe.mockResolvedValueOnce("allow")

    const ctx = buildCtx()
    const ev = makeEvaluation()
    await handlePermissionEvent(ev, ctx)

    // V2: auto-approval happens by mutating `ev.effect` BEFORE opencode
    // creates its TUI prompt — the handler never calls the SDK reply itself.
    expect(ev.effect).toBe("allow")
    expect(ctx.opencode.permission.reply).not.toHaveBeenCalled()
    expect(mockedRisky).not.toHaveBeenCalled()
  })

  it("leaves ev.effect='ask' when SAFE but user cancels (safe-path returns ask)", async () => {
    mockedClassify.mockResolvedValueOnce({
      verdict: "SAFE",
      reason: "read-only",
    })
    mockedSafe.mockResolvedValueOnce("ask")

    const ctx = buildCtx()
    const ev = makeEvaluation()
    await handlePermissionEvent(ev, ctx)

    expect(ev.effect).toBe("ask")
  })

  it("starts the risky-path in background when verdict is RISKY", async () => {
    mockedClassify.mockResolvedValueOnce({
      verdict: "RISKY",
      reason: "destructive",
    })

    const ctx = buildCtx()
    const ev = makeEvaluation({ resources: ["rm -rf /"] })
    await handlePermissionEvent(ev, ctx)

    // V2: the TUI prompt is left in place; the risky-path gets an injected
    // replier it can call on a notification-button click.
    expect(ev.effect).toBe("ask")
    expect(mockedRisky).toHaveBeenCalledTimes(1)
    const args = mockedRisky.mock.calls[0]?.[0]
    expect(args?.command).toBe("rm -rf /")
    expect(args?.reason).toBe("destructive")
    expect(typeof args?.reply).toBe("function")
  })

  it("resolves the pending request via the injected reply (RISKY path)", async () => {
    mockedClassify.mockResolvedValueOnce({
      verdict: "RISKY",
      reason: "destructive",
    })

    const ctx = buildCtx()
    vi.mocked(ctx.opencode.permission.list).mockResolvedValue([
      {
        id: "req_1",
        sessionID: "sess_root",
        action: "shell",
        resources: ["rm -rf /"],
      },
    ])
    const ev = makeEvaluation({ resources: ["rm -rf /"] })
    await handlePermissionEvent(ev, ctx)

    const reply = mockedRisky.mock.calls[0]?.[0]?.reply
    expect(reply).toBeTypeOf("function")
    await reply!("once")

    expect(ctx.opencode.permission.reply).toHaveBeenCalledWith({
      sessionID: "sess_root",
      requestID: "req_1",
      reply: "once",
    })
  })

  it("does not auto-resolve when the classifier fails (returns null)", async () => {
    mockedClassify.mockResolvedValueOnce(null)

    const ctx = buildCtx()
    const ev = makeEvaluation()
    await handlePermissionEvent(ev, ctx)

    // The plugin must never auto-approve/auto-reject on a classifier failure.
    expect(ev.effect).toBe("ask")
    expect(mockedSafe).not.toHaveBeenCalled()
    expect(mockedRisky).not.toHaveBeenCalled()
  })

  it("fires the failure notification when the classifier returns null", async () => {
    mockedClassify.mockResolvedValueOnce(null)

    const ctx = buildCtx()
    const ev = makeEvaluation({ resources: ["echo hi"] })
    await handlePermissionEvent(ev, ctx)

    expect(mockedFailureNotify).toHaveBeenCalledTimes(1)
    const args = mockedFailureNotify.mock.calls[0]?.[0]
    expect(args?.command).toBe("echo hi")
    // Failure class defaults to "error" when the classifier mock doesn't
    // invoke onFailure; the real classifier reports "timeout" vs "error".
    expect(["timeout", "error"]).toContain(args?.failureClass)
    expect(typeof args?.reply).toBe("function")
  })

  it("resolves the pending request via the injected reply (classifier-failure path)", async () => {
    mockedClassify.mockResolvedValueOnce(null)

    const ctx = buildCtx()
    vi.mocked(ctx.opencode.permission.list).mockResolvedValue([
      {
        id: "req_2",
        sessionID: "sess_root",
        action: "shell",
        resources: ["echo hi"],
      },
    ])
    const ev = makeEvaluation({ resources: ["echo hi"] })
    await handlePermissionEvent(ev, ctx)

    const reply = mockedFailureNotify.mock.calls[0]?.[0]?.reply
    expect(reply).toBeTypeOf("function")
    await reply!("reject")

    expect(ctx.opencode.permission.reply).toHaveBeenCalledWith({
      sessionID: "sess_root",
      requestID: "req_2",
      reply: "reject",
    })
  })

  it("does NOT fire the failure notification when notifyOnClassifierFailure is false", async () => {
    mockedClassify.mockResolvedValueOnce(null)

    const ctx = buildCtx({ config: { notifyOnClassifierFailure: false } })
    await handlePermissionEvent(makeEvaluation(), ctx)

    expect(mockedFailureNotify).not.toHaveBeenCalled()
  })

  it("does NOT fire the failure notification on a successful verdict", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    const ctx = buildCtx()
    await handlePermissionEvent(makeEvaluation(), ctx)

    expect(mockedFailureNotify).not.toHaveBeenCalled()
  })

  it("rate-limits a burst of failures into a single notification", async () => {
    // Shared rate limiter with a long cooldown across multiple permissions.
    const rl = new FailureNotifyRateLimiter({ cooldownMs: 60_000 })
    mockedClassify.mockResolvedValue(null)

    const ctx = buildCtx({ failureNotifyRateLimiter: rl })
    await handlePermissionEvent(
      makeEvaluation({ sessionID: "sess_1", resources: ["echo 1"] }),
      ctx,
    )
    await handlePermissionEvent(
      makeEvaluation({ sessionID: "sess_2", resources: ["echo 2"] }),
      ctx,
    )
    await handlePermissionEvent(
      makeEvaluation({ sessionID: "sess_3", resources: ["echo 3"] }),
      ctx,
    )

    // Only the first failure in the window actually notifies.
    expect(mockedFailureNotify).toHaveBeenCalledTimes(1)
  })

  it("does nothing when getSessionMessages throws", async () => {
    mockedGetSessionMessages.mockRejectedValueOnce(new Error("sdk explode"))

    const ctx = buildCtx()
    const ev = makeEvaluation()
    await handlePermissionEvent(ev, ctx)

    expect(mockedClassify).not.toHaveBeenCalled()
    expect(ev.effect).toBe("ask")
  })

  it("extracts the command from the resources array", async () => {
    mockedClassify.mockResolvedValueOnce({
      verdict: "SAFE",
      reason: "r",
    })
    mockedSafe.mockResolvedValueOnce("allow")

    const ctx = buildCtx()
    await handlePermissionEvent(
      makeEvaluation({ resources: ["echo hi"] }),
      ctx,
    )

    const args = mockedClassify.mock.calls[0]?.[0]
    expect(args?.command).toBe("echo hi")
  })

  it("classifies the FULL compound command, not just the first sub-command", async () => {
    // The V2 permission scanner splits a compound shell command (joined by
    // &&, ;, |, etc.) into its constituent sub-commands in `resources`.
    // Classifying only resources[0] would judge a different, often safer
    // command than what actually runs — a safe first segment could mask a
    // risky later one.
    mockedClassify.mockResolvedValueOnce({
      verdict: "SAFE",
      reason: "r",
    })
    mockedSafe.mockResolvedValueOnce("allow")

    const ctx = buildCtx()
    await handlePermissionEvent(
      makeEvaluation({ resources: ["git add .", 'git commit -m "wip"'] }),
      ctx,
    )

    const args = mockedClassify.mock.calls[0]?.[0]
    // Both sub-commands must be present in the classified subject.
    expect(args?.command).toContain("git add .")
    expect(args?.command).toContain('git commit -m "wip"')
  })

  it("does not let a safe first sub-command hide a risky later one", async () => {
    mockedClassify.mockResolvedValueOnce({
      verdict: "RISKY",
      reason: "r",
    })

    const ctx = buildCtx()
    await handlePermissionEvent(
      makeEvaluation({ resources: ["git status", "rm -rf /important"] }),
      ctx,
    )

    const args = mockedClassify.mock.calls[0]?.[0]
    expect(args?.command).toContain("git status")
    expect(args?.command).toContain("rm -rf /important")
  })

  it("classifies a single-element resources array as just that command", async () => {
    mockedClassify.mockResolvedValueOnce({
      verdict: "SAFE",
      reason: "r",
    })
    mockedSafe.mockResolvedValueOnce("allow")

    const ctx = buildCtx()
    await handlePermissionEvent(
      makeEvaluation({ resources: ["ls -la"] }),
      ctx,
    )

    const args = mockedClassify.mock.calls[0]?.[0]
    expect(args?.command).toBe("ls -la")
  })

  it("does nothing when resources is empty (no command to classify)", async () => {
    const ctx = buildCtx()
    const ev = makeEvaluation({ resources: [] })
    await handlePermissionEvent(ev, ctx)
    expect(mockedClassify).not.toHaveBeenCalled()
    expect(ev.effect).toBe("ask")
  })

  it("applies config.contextMessageCount when extracting user messages", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    // 5 user messages available; contextMessageCount=2 → classifier sees
    // only the last 2.
    mockedGetSessionMessages.mockResolvedValueOnce([
      userMessage("m1"),
      userMessage("m2"),
      userMessage("m3"),
      userMessage("m4"),
      userMessage("m5"),
    ])

    const ctx = buildCtx({ config: { contextMessageCount: 2 } })
    await handlePermissionEvent(makeEvaluation(), ctx)

    const classifyArgs = mockedClassify.mock.calls[0]?.[0]
    expect(classifyArgs?.userMessages).toEqual(["m4", "m5"])
  })

  it("uses the resolved classifier model (config override wins)", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    const ctx = buildCtx({
      config: { classifierModel: "anthropic/claude-haiku-4-5" },
    })
    await handlePermissionEvent(makeEvaluation(), ctx)

    const args = mockedClassify.mock.calls[0]?.[0]
    expect(args?.model).toEqual({
      providerID: "anthropic",
      modelID: "claude-haiku-4-5",
    })
  })

  it("forwards ctx.log to the classifier so failures are observable", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    const ctx = buildCtx()
    await handlePermissionEvent(makeEvaluation(), ctx)

    const args = mockedClassify.mock.calls[0]?.[0]
    expect(args?.log).toBe(ctx.log)
  })

  it("forwards config.classifierRetries to the classifier", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    const ctx = buildCtx()
    await handlePermissionEvent(makeEvaluation(), ctx)

    const args = mockedClassify.mock.calls[0]?.[0]
    // DEFAULT_CONFIG.classifierRetries is 1.
    expect(args?.retries).toBe(1)
  })

  it("forwards the generate domain to the classifier", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    const ctx = buildCtx()
    await handlePermissionEvent(makeEvaluation(), ctx)

    const args = mockedClassify.mock.calls[0]?.[0]
    expect(args?.generate).toBe(ctx.opencode.generate)
  })

  it("does nothing when no classifier model can be resolved", async () => {
    const ctx = buildCtx({
      sessionModel: undefined,
      config: { classifierModel: undefined },
    })
    const ev = makeEvaluation()
    await handlePermissionEvent(ev, ctx)
    expect(mockedClassify).not.toHaveBeenCalled()
    expect(ev.effect).toBe("ask")
  })

  it("injected reply is fail-closed when permission.list throws (TUI prompt remains)", async () => {
    // V2 SAFE path no longer calls the SDK reply; the only SDK interaction is
    // through the replier injected into the risky/failure paths. That replier
    // must never throw: any lookup failure leaves the TUI prompt in place.
    mockedClassify.mockResolvedValueOnce({
      verdict: "RISKY",
      reason: "r",
    })

    const ctx = buildCtx()
    vi.mocked(ctx.opencode.permission.list).mockRejectedValue(
      new Error("sdk boom"),
    )
    const ev = makeEvaluation()
    await handlePermissionEvent(ev, ctx)

    const reply = mockedRisky.mock.calls[0]?.[0]?.reply
    expect(reply).toBeTypeOf("function")
    await expect(reply!("once")).resolves.toBeUndefined()
    expect(ctx.opencode.permission.reply).not.toHaveBeenCalled()
    expect(ev.effect).toBe("ask")
  })

  // --- V2 evaluate-hook effect (set before the TUI prompt exists) --------

  it("sets ev.effect='allow' on a SAFE-allow verdict without replying to the SDK", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    const ctx = buildCtx()
    const ev = makeEvaluation()
    await handlePermissionEvent(ev, ctx)

    expect(ev.effect).toBe("allow")
    expect(ctx.opencode.permission.reply).not.toHaveBeenCalled()
  })

  it("leaves ev.effect='ask' on a SAFE verdict when the user cancels", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("ask")

    const ctx = buildCtx()
    const ev = makeEvaluation()
    await handlePermissionEvent(ev, ctx)

    expect(ev.effect).toBe("ask")
    expect(ctx.opencode.permission.reply).not.toHaveBeenCalled()
  })

  it("leaves ev.effect='ask' on a RISKY verdict and kicks off the risky path", async () => {
    mockedClassify.mockResolvedValueOnce({
      verdict: "RISKY",
      reason: "destructive",
    })

    const ctx = buildCtx()
    const ev = makeEvaluation({ resources: ["rm -rf /"] })
    await handlePermissionEvent(ev, ctx)

    // TUI prompt should still be shown; notification runs alongside.
    expect(ev.effect).toBe("ask")
    expect(mockedRisky).toHaveBeenCalledTimes(1)
    expect(ctx.opencode.permission.reply).not.toHaveBeenCalled()
  })

  it("leaves ev.effect='ask' when the classifier fails (fail closed)", async () => {
    mockedClassify.mockResolvedValueOnce(null)

    const ctx = buildCtx()
    const ev = makeEvaluation()
    await handlePermissionEvent(ev, ctx)

    expect(ev.effect).toBe("ask")
    expect(ctx.opencode.permission.reply).not.toHaveBeenCalled()
  })

  // --- session-model fallback from latest assistant message -------------
  //
  // When the `config` hook hasn't surfaced `ctx.sessionModel` (e.g. the
  // hook didn't fire, or opencode's runtime Config uses different field
  // names), the handler falls back to the latest assistant message's
  // model in the session's transcript.

  it("uses assistant-message model fallback when ctx.sessionModel is undefined", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    // Transcript contains an assistant with a model; ctx.sessionModel
    // is undefined; no classifier override.
    mockedGetSessionMessages.mockResolvedValueOnce([
      userMessage("help me out"),
      assistantMessage({ providerID: "openai", id: "gpt-5-codex" }),
      userMessage("thanks"),
    ])

    const ctx = buildCtx({ sessionModel: undefined })
    await handlePermissionEvent(makeEvaluation(), ctx)

    expect(mockedClassify).toHaveBeenCalledTimes(1)
    const classifyArgs = mockedClassify.mock.calls[0]?.[0]
    // openai has a provider default (gpt-5.4-mini), which the resolver
    // prefers over the fallback's raw modelID.
    expect(classifyArgs?.model.providerID).toBe("openai")
  })

  it("still skips with 'no classifier model' when every source fails", async () => {
    // No ctx.sessionModel, no config override, transcript has no
    // assistant with a model → resolver returns null → handler skips.
    mockedGetSessionMessages.mockResolvedValueOnce([userMessage("just me")])
    const ctx = buildCtx({ sessionModel: undefined })

    const ev = makeEvaluation()
    await handlePermissionEvent(ev, ctx)

    expect(mockedClassify).not.toHaveBeenCalled()
    expect(ev.effect).toBe("ask")
  })

  it("ctx.sessionModel takes precedence over the assistant-message fallback", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    // Assistant in transcript says openai/gpt-5, but ctx.sessionModel says
    // anthropic/claude-sonnet. The explicit ctx value wins.
    mockedGetSessionMessages.mockResolvedValueOnce([
      userMessage("hi"),
      assistantMessage({ providerID: "openai", id: "gpt-5" }),
    ])

    const ctx = buildCtx({
      sessionModel: { providerID: "anthropic", modelID: "claude-sonnet-4-5" },
    })
    await handlePermissionEvent(makeEvaluation(), ctx)

    const classifyArgs = mockedClassify.mock.calls[0]?.[0]
    expect(classifyArgs?.model.providerID).toBe("anthropic")
  })

  it("config classifierModel override takes precedence over both session and fallback", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    mockedGetSessionMessages.mockResolvedValueOnce([
      userMessage("hi"),
      assistantMessage({ providerID: "openai", id: "gpt-5" }),
    ])

    const ctx = buildCtx({
      config: { classifierModel: "anthropic/claude-haiku-4-5" },
      sessionModel: { providerID: "openai", modelID: "gpt-4o" },
    })
    await handlePermissionEvent(makeEvaluation(), ctx)

    const classifyArgs = mockedClassify.mock.calls[0]?.[0]
    expect(classifyArgs?.model).toEqual({
      providerID: "anthropic",
      modelID: "claude-haiku-4-5",
    })
  })

  // --- subagent handling ------------------------------------------------
  //
  // When a shell permission fires inside a subagent session, the handler
  // must resolve the session's root (via resolveRootSessionID) and fetch
  // user messages from THERE, not from the subagent session — whose
  // "user" role entries are actually the dispatching agent's prompts.
  // Any failure to resolve a root is fail-closed: the handler skips
  // classification and leaves the TUI prompt alone for the user.

  it("fetches messages from the ROOT session when permission fires in a subagent", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    // Simulate: permission arrives with the subagent's sessionID; resolver
    // walks up and returns the root sessionID.
    mockedResolveRoot.mockImplementationOnce(async () => "sess_root")

    const ctx = buildCtx()
    await handlePermissionEvent(
      makeEvaluation({ sessionID: "sess_subagent" }),
      ctx,
    )

    expect(mockedResolveRoot).toHaveBeenCalledTimes(1)
    expect(mockedResolveRoot.mock.calls[0]?.[1]).toBe("sess_subagent")

    // Messages must come from the ROOT session, not the subagent.
    expect(mockedGetSessionMessages).toHaveBeenCalledTimes(1)
    expect(mockedGetSessionMessages.mock.calls[0]?.[1]).toBe("sess_root")
  })

  it("fetches messages from the permission's own sessionID for a root session", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    // Default beforeEach behaviour: resolver returns the input sessionID
    // unchanged (i.e. already a root). No override needed.

    const ctx = buildCtx()
    await handlePermissionEvent(
      makeEvaluation({ sessionID: "sess_root_only" }),
      ctx,
    )

    expect(mockedGetSessionMessages).toHaveBeenCalledTimes(1)
    expect(mockedGetSessionMessages.mock.calls[0]?.[1]).toBe("sess_root_only")
  })

  it("skips classification and does not call getSessionMessages when resolver returns null", async () => {
    // Resolver fail-closed: subagent's chain couldn't be verified.
    mockedResolveRoot.mockImplementationOnce(async () => null)

    const ctx = buildCtx()
    const ev = makeEvaluation({ sessionID: "sess_subagent" })
    await handlePermissionEvent(ev, ctx)

    // No message fetch, no classification, no response.
    expect(mockedGetSessionMessages).not.toHaveBeenCalled()
    expect(mockedClassify).not.toHaveBeenCalled()
    expect(mockedSafe).not.toHaveBeenCalled()
    expect(mockedRisky).not.toHaveBeenCalled()
    expect(ev.effect).toBe("ask")
  })

  it("keeps the permission's ORIGINAL sessionID as the classifier's parentSessionID", async () => {
    // Even when the resolver discovers a different root, the classifier is
    // told which session the permission actually fired in (the subagent's),
    // so attribution stays exact. (User messages still come from the root.)
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")
    mockedResolveRoot.mockImplementationOnce(async () => "sess_root")

    const ctx = buildCtx()
    await handlePermissionEvent(
      makeEvaluation({ sessionID: "sess_subagent" }),
      ctx,
    )

    const classifyArgs = mockedClassify.mock.calls[0]?.[0]
    expect(classifyArgs?.parentSessionID).toBe("sess_subagent")
  })

  it("classifier sees the ROOT session's user messages, not the subagent's", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")
    mockedResolveRoot.mockImplementationOnce(async () => "sess_root")

    // Tell getSessionMessages to return DIFFERENT message sets depending
    // on which sessionID is requested. The handler should request
    // `sess_root` (the resolved root), so the classifier must see the
    // root's human messages — not the subagent's dispatch prompt.
    mockedGetSessionMessages.mockImplementation(async (_session, id) => {
      if (id === "sess_root") return [userMessage("the real human said this")]
      return [userMessage("dispatching agent's prompt to subagent")]
    })

    const ctx = buildCtx()
    await handlePermissionEvent(
      makeEvaluation({ sessionID: "sess_subagent" }),
      ctx,
    )

    const classifyArgs = mockedClassify.mock.calls[0]?.[0]
    expect(classifyArgs?.userMessages).toEqual(["the real human said this"])
  })

  // --- repo context wiring ----------------------------------------------

  it("forwards repo context from getRepoContext to classifyCommand", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    const repoCtx = {
      branch: "feat/x",
      openPR: { number: 7, title: "Test", baseBranch: "main" },
    }
    const ctx = buildCtx({
      getRepoContext: vi.fn(async () => repoCtx),
    })

    await handlePermissionEvent(makeEvaluation(), ctx)

    const classifyArgs = mockedClassify.mock.calls[0]?.[0]
    expect(classifyArgs?.repoContext).toEqual(repoCtx)
  })

  it("forwards null repo context when fetcher returns null", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    const ctx = buildCtx({
      getRepoContext: vi.fn(async () => null),
    })

    await handlePermissionEvent(makeEvaluation(), ctx)

    const classifyArgs = mockedClassify.mock.calls[0]?.[0]
    expect(classifyArgs?.repoContext).toBeNull()
  })

  it("forwards null repo context when getRepoContext throws", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    const ctx = buildCtx({
      getRepoContext: vi.fn(async () => {
        throw new Error("boom")
      }),
    })

    await handlePermissionEvent(makeEvaluation(), ctx)

    const classifyArgs = mockedClassify.mock.calls[0]?.[0]
    expect(classifyArgs?.repoContext).toBeNull()
  })

  it("works without a getRepoContext (backward-compat with older ctx)", async () => {
    mockedClassify.mockResolvedValueOnce({ verdict: "SAFE", reason: "r" })
    mockedSafe.mockResolvedValueOnce("allow")

    const ctx = buildCtx({ getRepoContext: undefined })
    await handlePermissionEvent(makeEvaluation(), ctx)

    const classifyArgs = mockedClassify.mock.calls[0]?.[0]
    expect(classifyArgs?.repoContext).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// external_directory permission handling
// ---------------------------------------------------------------------------

const DEFAULT_DIR_PATH = "/Users/jacob/Documents/GitHub/premind/*"

/** Build a V2 external_directory permission evaluation. */
function dirEvaluation(
  path = DEFAULT_DIR_PATH,
  overrides: Partial<PermissionEvaluation> = {},
): PermissionEvaluation {
  return makeEvaluation({
    action: "external_directory",
    resources: [path],
    ...overrides,
  })
}

describe("handlePermissionEvent (external_directory)", () => {
  beforeEach(() => {
    // Default: root session resolves to itself, one user message.
    mockedResolveRoot.mockResolvedValue("sess_root")
    mockedGetSessionMessages.mockResolvedValue([
      userMessage("please review the premind project"),
    ])
    // Default: classifyDirectory returns SAFE.
    mockedClassifyDirectory.mockResolvedValue({
      verdict: "SAFE",
      reason: "user asked for premind",
    })
    mockedSafe.mockResolvedValue("allow")
  })

  it("calls classifyDirectory (not classifyCommand) for external_directory", async () => {
    const ctx = buildCtx()
    await handlePermissionEvent(dirEvaluation(), ctx)
    expect(mockedClassifyDirectory).toHaveBeenCalledTimes(1)
    expect(mockedClassify).not.toHaveBeenCalled()
  })

  it("passes the directory path as subject to classifyDirectory", async () => {
    const ctx = buildCtx()
    await handlePermissionEvent(
      dirEvaluation("/Users/jacob/Documents/GitHub/premind/*"),
      ctx,
    )
    const args = mockedClassifyDirectory.mock.calls[0]?.[0]
    expect(args?.path).toBe("/Users/jacob/Documents/GitHub/premind/*")
  })

  it("auto-approves when classifyDirectory returns SAFE and safe-path allows", async () => {
    const ctx = buildCtx()
    const ev = dirEvaluation()
    await handlePermissionEvent(ev, ctx)
    expect(mockedSafe).toHaveBeenCalledTimes(1)
    expect(ev.effect).toBe("allow")
  })

  it("escalates via risky-path when classifyDirectory returns RISKY", async () => {
    mockedClassifyDirectory.mockResolvedValue({
      verdict: "RISKY",
      reason: "sensitive path",
    })
    const ctx = buildCtx()
    const ev = dirEvaluation()
    await handlePermissionEvent(ev, ctx)
    expect(mockedRisky).toHaveBeenCalledTimes(1)
    expect(ev.effect).toBe("ask")
  })

  it("does not call classifyDirectory on cache hit; still runs safe-path", async () => {
    const path = "/Users/jacob/Documents/GitHub/premind/*"
    const ctx = buildCtx()

    // Pre-populate cache as SAFE.
    ctx.directoryVerdictCache.set(
      DirectoryVerdictCache.keyFor([path]),
      { verdict: "SAFE", reason: "cached" },
      60_000,
    )

    await handlePermissionEvent(dirEvaluation(path), ctx)

    expect(mockedClassifyDirectory).not.toHaveBeenCalled()
    expect(mockedSafe).toHaveBeenCalledTimes(1)
  })

  it("populates the cache after a fresh SAFE verdict", async () => {
    const path = "/Users/jacob/Documents/GitHub/premind/*"
    const ctx = buildCtx()

    await handlePermissionEvent(dirEvaluation(path), ctx)

    const cacheKey = DirectoryVerdictCache.keyFor([path])
    const cached = ctx.directoryVerdictCache.get(cacheKey)
    expect(cached).not.toBeNull()
    expect(cached?.verdict.verdict).toBe("SAFE")
  })

  it("does not populate the cache after a RISKY verdict", async () => {
    mockedClassifyDirectory.mockResolvedValue({
      verdict: "RISKY",
      reason: "sensitive",
    })
    const path = "/Users/jacob/Documents/GitHub/premind/*"
    const ctx = buildCtx()

    await handlePermissionEvent(dirEvaluation(path), ctx)

    const cacheKey = DirectoryVerdictCache.keyFor([path])
    expect(ctx.directoryVerdictCache.get(cacheKey)).toBeNull()
  })

  it("skips external_directory when externalDirectoryEnabled is false", async () => {
    const ctx = buildCtx({ config: { externalDirectoryEnabled: false } })
    const ev = dirEvaluation()

    await handlePermissionEvent(ev, ctx)

    expect(mockedClassifyDirectory).not.toHaveBeenCalled()
    expect(mockedClassify).not.toHaveBeenCalled()
    expect(mockedSafe).not.toHaveBeenCalled()
    expect(ev.effect).toBe("ask")
  })

  it("skips external_directory (and all others) when enabled is false", async () => {
    const ctx = buildCtx({ config: { enabled: false } })
    await handlePermissionEvent(dirEvaluation(), ctx)
    expect(mockedClassifyDirectory).not.toHaveBeenCalled()
  })

  it("falls back to TUI prompt when classifyDirectory returns null", async () => {
    mockedClassifyDirectory.mockResolvedValue(null)
    const ctx = buildCtx()
    const ev = dirEvaluation()
    await handlePermissionEvent(ev, ctx)
    expect(ev.effect).toBe("ask")
    expect(mockedSafe).not.toHaveBeenCalled()
  })

  it("leaves TUI prompt when user cancels the safe-path countdown", async () => {
    mockedSafe.mockResolvedValue("ask")
    const ctx = buildCtx()
    const ev = dirEvaluation()
    await handlePermissionEvent(ev, ctx)
    expect(ev.effect).toBe("ask")
  })
})

describe("approval history wiring", () => {
  beforeEach(() => {
    // The wiring tests always use sess_test as the root; pin it
    // explicitly rather than relying on `mockReset` leaving the mock as
    // `undefined` (which would propagate as the rootSessionID and break
    // the priorApprovals lookup test).
    mockedResolveRoot.mockResolvedValue("sess_test")
  })

  it("seeds a pending subject when a shell permission first fires", async () => {
    const pending = new PendingSubjectsMap()
    mockedClassify.mockResolvedValue({ verdict: "RISKY", reason: "test" })
    mockedSafe.mockResolvedValue("ask")
    const ctx = buildCtx({ pendingSubjects: pending })

    const ev = makeEvaluation({ sessionID: "sess_test", resources: ["ls -la"] })
    await handlePermissionEvent(ev, ctx)

    // V2 keys pending entries by evaluationKey(ev) — the evaluate event
    // carries no permission ID.
    const taken = pending.take(evaluationKey(ev))
    expect(taken).not.toBeNull()
    expect(taken?.subject).toBe("ls -la")
    expect(taken?.subjectLabel).toBe("command")
    expect(taken?.classifierVerdict).toBe("RISKY")
  })

  it("seeds with subjectLabel='path' for external_directory permissions", async () => {
    const pending = new PendingSubjectsMap()
    mockedClassifyDirectory.mockResolvedValue({
      verdict: "RISKY",
      reason: "no context",
    })
    const ctx = buildCtx({ pendingSubjects: pending })

    const ev = makeEvaluation({
      sessionID: "sess_test",
      action: "external_directory",
      resources: ["/Users/jacob/Documents/GitHub/other/*"],
    })
    await handlePermissionEvent(ev, ctx)

    const taken = pending.take(evaluationKey(ev))
    expect(taken?.subjectLabel).toBe("path")
  })

  it("marks the pending subject autoApproved when SAFE path resolves to allow", async () => {
    const pending = new PendingSubjectsMap()
    mockedClassify.mockResolvedValue({ verdict: "SAFE", reason: "read-only" })
    mockedSafe.mockResolvedValue("allow")
    const ctx = buildCtx({ pendingSubjects: pending })

    const ev = makeEvaluation({ sessionID: "sess_test", resources: ["git status"] })
    await handlePermissionEvent(ev, ctx)

    const taken = pending.take(evaluationKey(ev))
    expect(taken?.autoApproved).toBe(true)
  })

  it("does NOT mark autoApproved when SAFE path resolves to ask (user cancelled)", async () => {
    const pending = new PendingSubjectsMap()
    mockedClassify.mockResolvedValue({ verdict: "SAFE", reason: "read-only" })
    mockedSafe.mockResolvedValue("ask")
    const ctx = buildCtx({ pendingSubjects: pending })

    const ev = makeEvaluation({ sessionID: "sess_test", resources: ["git status"] })
    await handlePermissionEvent(ev, ctx)

    const taken = pending.take(evaluationKey(ev))
    expect(taken?.autoApproved).toBe(false)
  })

  it("passes priorApprovals from the store into classifyCommand", async () => {
    const history = new ApprovalHistoryStore()
    history.record("sess_test", {
      subject: "gh pr comment 1 -b 'a'",
      subjectLabel: "command",
      response: "once",
      classifierVerdict: "RISKY",
      classifierReason: "PR not matched",
      timestamp: 1_000,
    })
    mockedClassify.mockResolvedValue({ verdict: "SAFE", reason: "stub" })
    mockedSafe.mockResolvedValue("allow")
    const ctx = buildCtx({ approvalHistory: history })

    await handlePermissionEvent(
      makeEvaluation({
        sessionID: "sess_test",
        resources: ["gh pr comment 1 -b 'b'"],
      }),
      ctx,
    )

    expect(mockedClassify).toHaveBeenCalledTimes(1)
    const callArgs = mockedClassify.mock.calls[0]?.[0]
    expect(callArgs?.priorApprovals?.length).toBe(1)
    expect(callArgs?.priorApprovals?.[0]?.subject).toBe(
      "gh pr comment 1 -b 'a'",
    )
  })

  it("passes empty priorApprovals when approvalHistoryEnabled is false", async () => {
    const history = new ApprovalHistoryStore()
    history.record("sess_test", {
      subject: "ls",
      subjectLabel: "command",
      response: "once",
      classifierVerdict: "SAFE",
      classifierReason: "x",
      timestamp: 1_000,
    })
    mockedClassify.mockResolvedValue({ verdict: "SAFE", reason: "stub" })
    mockedSafe.mockResolvedValue("allow")
    const ctx = buildCtx({
      approvalHistory: history,
      config: { approvalHistoryEnabled: false },
    })

    await handlePermissionEvent(
      makeEvaluation({ sessionID: "sess_test", resources: ["ls"] }),
      ctx,
    )

    const callArgs = mockedClassify.mock.calls[0]?.[0]
    expect(callArgs?.priorApprovals).toEqual([])
  })

  it("updates the pending subject's verdict after the classifier returns", async () => {
    const pending = new PendingSubjectsMap()
    mockedClassify.mockResolvedValue({
      verdict: "SAFE",
      reason: "looks fine",
    })
    mockedSafe.mockResolvedValue("ask")
    const ctx = buildCtx({ pendingSubjects: pending })

    const ev = makeEvaluation({ sessionID: "sess_test", resources: ["ls"] })
    await handlePermissionEvent(ev, ctx)

    const taken = pending.take(evaluationKey(ev))
    expect(taken?.classifierVerdict).toBe("SAFE")
    expect(taken?.classifierReason).toBe("looks fine")
  })
})

describe("dual repo context wiring", () => {
  beforeEach(() => {
    mockedResolveRoot.mockResolvedValue("sess_test")
  })

  it("passes the dual repo context through to the classifier", async () => {
    mockedClassify.mockResolvedValue({ verdict: "SAFE", reason: "stub" })
    mockedSafe.mockResolvedValue("allow")
    const ctx = buildCtx({
      getRepoContext: async () => ({
        pinned: {
          branch: "feat/x",
          openPR: { number: 99, title: "P", baseBranch: "main" },
        },
        current: {
          branch: "feat/x",
          openPR: { number: 99, title: "P", baseBranch: "main" },
        },
      }),
    })

    await handlePermissionEvent(
      makeEvaluation({
        sessionID: "sess_test",
        resources: ["gh pr comment 99 -b 'reply'"],
      }),
      ctx,
    )

    expect(mockedClassify).toHaveBeenCalledTimes(1)
    const args = mockedClassify.mock.calls[0]?.[0]
    expect(args?.repoContext).toMatchObject({
      pinned: { branch: "feat/x" },
      current: { branch: "feat/x" },
    })
  })
})
