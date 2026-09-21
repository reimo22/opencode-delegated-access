import { describe, it, expect, vi, beforeEach } from "vitest"

// Keep `evaluationKey` real; stub only the handler so we can observe dispatch.
vi.mock("./permission/handler.ts", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./permission/handler.ts")>()
  return { ...actual, handlePermissionEvent: vi.fn() }
})

// Never write to the real user log file from a test run.
vi.mock("./log.ts", () => ({
  createLogger: () => ({
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  }),
}))

import DelegatedAccess from "./index.ts"
import {
  handlePermissionEvent,
  evaluationKey,
  type HandlerContext,
} from "./permission/handler.ts"
import { makeEvaluation, makePluginContext } from "./testing/v2-fixtures.ts"

const mockedHandle = vi.mocked(handlePermissionEvent)

/** Root session every test records history against. */
const ROOT = "sess_root"

/** The evaluate event whose pending subject the replied path looks for. */
const STATUS_EVALUATION = {
  sessionID: ROOT,
  action: "shell",
  resources: ["git status"],
}
const STATUS_KEY = evaluationKey(STATUS_EVALUATION)

/** The permission request the plugin's `permission.list` reports back. */
const STATUS_REQUEST = { id: "perm_1", action: "shell", resources: ["git status"] }

/**
 * Let the plugin's background event loop drain everything queued so far.
 *
 * The loop is pure promise chaining with no timers or I/O, and the fixture
 * stream drains its queue in FIFO order before reporting `done`, so a closed
 * stream plus a couple of macrotask turns settles every pushed event. This is
 * what makes the "nothing was recorded" assertions meaningful instead of
 * passing trivially at t=0.
 */
async function drainEventLoop() {
  for (let i = 0; i < 3; i++) {
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
}

beforeEach(() => {
  mockedHandle.mockReset()
  mockedHandle.mockImplementation(async () => {
    // Default: do nothing.
  })
})

/**
 * Run the V2 plugin's `setup` against a fake context and hand back the hooks
 * it registered. `setup` is a plain method on the exported plugin object —
 * there is no callable factory in V2.
 */
async function setupPlugin(options?: Record<string, unknown>) {
  const { ctx, session, generate, permission, event } = makePluginContext({
    options,
  })

  await DelegatedAccess.setup(ctx as never)

  const evaluate = permission.hook.mock.calls.find(
    (call) => call[0] === "evaluate",
  )?.[1] as ((ev: unknown) => Promise<void>) | undefined

  return { evaluate, session, generate, permission, event, ctx }
}

/** One pending subject, as the evaluate path would have seeded it. */
type PendingSpec = {
  sessionID: string
  action: string
  resources: string[]
  entry: {
    rootSessionID: string
    subject: string
    subjectLabel: "path" | "command"
    classifierVerdict: "SAFE" | "RISKY" | null
    classifierReason: string | null
    autoApproved: boolean
  }
}

type RepliedEvent = { data: unknown; type?: string }

/**
 * Drive the whole `permission.replied` path the way production does: seed
 * pending subjects from inside the evaluate hook, then push events into the
 * plugin's background event stream and let it settle.
 *
 * `handlePermissionReplied` is module-private, so this is the only honest way
 * to reach it.
 */
async function driveReplied(args: {
  options?: Record<string, unknown>
  /** What `ctx.permission.list` reports for the session. */
  requests?: Array<{ id: string; action: string; resources: string[] }>
  pending?: PendingSpec[]
  events: RepliedEvent[]
}) {
  const { evaluate, permission, event } = await setupPlugin(args.options)
  if (args.requests) {
    permission.list.mockResolvedValue(args.requests as never)
  }

  let ctx: HandlerContext | undefined
  mockedHandle.mockImplementationOnce(async (_ev, handlerCtx) => {
    ctx = handlerCtx
    for (const spec of args.pending ?? []) {
      handlerCtx.pendingSubjects.set(
        evaluationKey({
          sessionID: spec.sessionID,
          action: spec.action,
          resources: spec.resources,
        }),
        spec.entry,
      )
    }
  })

  const first = args.pending?.[0]
  await evaluate!(
    makeEvaluation({
      sessionID: first?.sessionID ?? ROOT,
      action: first?.action ?? "shell",
      resources: first?.resources ?? ["git status"],
    }),
  )

  for (const e of args.events) {
    event.push({ type: e.type ?? "permission.replied", data: e.data })
  }
  event.close()
  await drainEventLoop()

  return ctx!
}

/** A RISKY pending subject for `git status`, the common case here. */
function riskyPending(
  args: {
    subject?: string
    resources?: string[]
    entry?: Partial<PendingSpec["entry"]>
  } = {},
): PendingSpec {
  const resources = args.resources ?? ["git status"]
  return {
    sessionID: ROOT,
    action: "shell",
    resources,
    entry: {
      rootSessionID: ROOT,
      subject: args.subject ?? "rm -rf /tmp/junk",
      subjectLabel: "command",
      classifierVerdict: "RISKY",
      classifierReason: "rm outside project",
      autoApproved: false,
      ...args.entry,
    },
  }
}

describe("DelegatedAccess setup — V2 hook registration", () => {
  it("registers the permission evaluate hook and the event stream", async () => {
    const { evaluate, event } = await setupPlugin()

    expect(typeof evaluate).toBe("function")
    expect(event.subscribe).toHaveBeenCalledTimes(1)
  })

  it("dispatches each evaluation to the handler with a handler context", async () => {
    const { evaluate, generate } = await setupPlugin()
    const ev = makeEvaluation()

    await evaluate!(ev)

    expect(mockedHandle).toHaveBeenCalledTimes(1)
    expect(mockedHandle.mock.calls[0]?.[0]).toBe(ev)
    const ctx = mockedHandle.mock.calls[0]?.[1]
    expect(typeof ctx!.log.info).toBe("function")
    expect(ctx!.pendingSubjects).toBeDefined()
    // `generate` is behind an `as unknown as` cast in index.ts, so the
    // compiler can't catch its removal. Without it the classifier fails
    // closed on every permission — assert the wiring explicitly.
    expect(ctx!.opencode.generate).toBe(generate)
  })

  it("swallows handler exceptions and leaves effect untouched (fail-closed)", async () => {
    const { evaluate } = await setupPlugin()
    mockedHandle.mockImplementation(async () => {
      throw new Error("unexpected boom")
    })

    const ev = makeEvaluation()

    await expect(evaluate!(ev)).resolves.toBeUndefined()
    // Untouched "ask" is the whole point: an exception must never
    // auto-approve the command.
    expect(ev.effect).toBe("ask")
  })
})

describe("plugin options → handler config", () => {
  async function configFor(options?: Record<string, unknown>) {
    const { evaluate } = await setupPlugin(options)
    await evaluate!(makeEvaluation())
    return mockedHandle.mock.calls[0]?.[1]
  }

  it("uses defaults when no options are supplied", async () => {
    const ctx = await configFor()
    expect(ctx?.config.enabled).toBe(true)
    expect(ctx?.config.contextMessageCount).toBe(3)
    expect(ctx?.sessionModel).toBeUndefined()
  })

  it("uses defaults when options is an empty object", async () => {
    const ctx = await configFor({})
    expect(ctx?.config.enabled).toBe(true)
    expect(ctx?.config.contextMessageCount).toBe(3)
  })

  it("applies tuple-form options", async () => {
    const ctx = await configFor({
      enabled: false,
      contextMessageCount: 5,
      safeCountdownMs: 0,
    })
    expect(ctx?.config.enabled).toBe(false)
    expect(ctx?.config.contextMessageCount).toBe(5)
    expect(ctx?.config.safeCountdownMs).toBe(0)
  })

  it("falls back to defaults when options are invalid", async () => {
    const ctx = await configFor({ enabled: "not a boolean" })
    expect(ctx?.config.enabled).toBe(true)
    expect(ctx?.config.contextMessageCount).toBe(3)
  })

  it("parses the classifier fallback model from options.model", async () => {
    const ctx = await configFor({ model: "anthropic/claude-sonnet-4-5" })
    expect(ctx?.sessionModel).toEqual({
      providerID: "anthropic",
      modelID: "claude-sonnet-4-5",
    })
  })
})

describe("repo context wiring", () => {
  it("exposes getRepoContext returning a { pinned, current } snapshot", async () => {
    const { evaluate } = await setupPlugin()
    await evaluate!(makeEvaluation())
    const ctx = mockedHandle.mock.calls[0]?.[1]

    expect(typeof ctx?.getRepoContext).toBe("function")

    const dual = await ctx!.getRepoContext!()
    expect(dual).not.toBeNull()
    expect("pinned" in dual!).toBe(true)
    expect("current" in dual!).toBe(true)
  })
})

describe("permission.replied → approval history", () => {
  it("records a human approval reported with the V2 runtime field names (requestID/reply)", async () => {
    const ctx = await driveReplied({
      requests: [STATUS_REQUEST],
      pending: [riskyPending()],
      events: [{ data: { sessionID: ROOT, requestID: "perm_1", reply: "once" } }],
    })

    const entries = ctx.approvalHistory.recent(ROOT, 10)
    expect(entries.length).toBe(1)
    expect(entries[0]?.subject).toBe("rm -rf /tmp/junk")
    expect(entries[0]?.response).toBe("once")
    expect(entries[0]?.classifierVerdict).toBe("RISKY")
    expect(Number.isFinite(entries[0]?.timestamp)).toBe(true)
  })

  it("records a rejection identically", async () => {
    const ctx = await driveReplied({
      requests: [STATUS_REQUEST],
      pending: [riskyPending()],
      events: [{ data: { sessionID: ROOT, requestID: "perm_1", reply: "reject" } }],
    })

    expect(ctx.approvalHistory.recent(ROOT, 10)[0]?.response).toBe("reject")
  })

  it("prefers the V2 event keys (requestID/reply) over the legacy V1 names", async () => {
    // The observable difference is WHICH request the plugin looks up, so give
    // the two id shapes different resources and see which one wins.
    const ctx = await driveReplied({
      requests: [
        { id: "perm_v2", action: "shell", resources: ["v2"] },
        { id: "perm_v1", action: "shell", resources: ["v1"] },
      ],
      pending: [
        riskyPending({ subject: "v2 subject", resources: ["v2"] }),
        riskyPending({ subject: "v1 subject", resources: ["v1"] }),
      ],
      events: [
        {
          data: {
            sessionID: ROOT,
            requestID: "perm_v2",
            reply: "once",
            permissionID: "perm_v1",
            response: "reject",
          },
        },
      ],
    })

    const entries = ctx.approvalHistory.recent(ROOT, 10)
    expect(entries.length).toBe(1)
    expect(entries[0]?.subject).toBe("v2 subject")
    expect(entries[0]?.response).toBe("once")
  })

  it("accepts the legacy V1 names when the V2 names are absent", async () => {
    const ctx = await driveReplied({
      requests: [STATUS_REQUEST],
      pending: [riskyPending()],
      events: [
        { data: { sessionID: ROOT, permissionID: "perm_1", response: "reject" } },
      ],
    })

    const entries = ctx.approvalHistory.recent(ROOT, 10)
    expect(entries.length).toBe(1)
    expect(entries[0]?.response).toBe("reject")
  })

  it("skips our own auto-approvals so history stays pure human signal", async () => {
    const ctx = await driveReplied({
      requests: [STATUS_REQUEST],
      pending: [riskyPending({ entry: { autoApproved: true } })],
      events: [{ data: { sessionID: ROOT, requestID: "perm_1", reply: "once" } }],
    })

    expect(ctx.approvalHistory.recent(ROOT, 10)).toEqual([])
    // The handler reached the pending subject and consumed it — proof the
    // event was processed rather than ignored.
    expect(ctx.pendingSubjects.take(STATUS_KEY)).toBeNull()
  })

  it("records a placeholder verdict when the human resolved before the classifier finished", async () => {
    const ctx = await driveReplied({
      requests: [STATUS_REQUEST],
      pending: [riskyPending({ entry: { classifierVerdict: null, classifierReason: null } })],
      events: [{ data: { sessionID: ROOT, requestID: "perm_1", reply: "once" } }],
    })

    const entry = ctx.approvalHistory.recent(ROOT, 10)[0]
    expect(entry?.classifierVerdict).toBe("RISKY")
    expect(entry?.classifierReason).toMatch(/classifier did not complete/)
  })

  it("does not record for an unrecognised reply value, but still consumes the pending subject", async () => {
    const ctx = await driveReplied({
      requests: [STATUS_REQUEST],
      pending: [riskyPending()],
      events: [{ data: { sessionID: ROOT, requestID: "perm_1", reply: "maybe" } }],
    })

    expect(ctx.approvalHistory.recent(ROOT, 10)).toEqual([])
    expect(ctx.pendingSubjects.take(STATUS_KEY)).toBeNull()
  })

  it("is a no-op when no pending subject exists for the request", async () => {
    const ctx = await driveReplied({
      requests: [STATUS_REQUEST],
      events: [{ data: { sessionID: ROOT, requestID: "perm_unknown", reply: "once" } }],
      // A pending entry for a *different* subject proves the loop ran.
      pending: [riskyPending()],
    })

    expect(ctx.approvalHistory.recent(ROOT, 10)).toEqual([])
  })

  it("honours approvalHistoryEnabled=false without consuming the pending entry", async () => {
    const ctx = await driveReplied({
      options: { approvalHistoryEnabled: false },
      requests: [STATUS_REQUEST],
      pending: [riskyPending()],
      events: [{ data: { sessionID: ROOT, requestID: "perm_1", reply: "once" } }],
    })

    expect(ctx.approvalHistory.recent(ROOT, 10)).toEqual([])
    // Disabled means disabled: the entry is left for a later decision.
    expect(ctx.pendingSubjects.take(STATUS_KEY)).not.toBeNull()
  })

  it.each([
    ["a missing sessionID", { requestID: "perm_1", reply: "once" }],
    ["no recognisable request id", { sessionID: ROOT, reply: "once" }],
    ["no recognisable reply", { sessionID: ROOT, requestID: "perm_1" }],
    ["a non-object payload", "nonsense"],
    ["a null payload", null],
  ])("ignores %s and keeps the event loop alive", async (_label, data) => {
    // The second, well-formed event proves the malformed one was skipped
    // rather than crashing the background loop.
    const ctx = await driveReplied({
      requests: [
        STATUS_REQUEST,
        { id: "perm_2", action: "shell", resources: ["ls"] },
      ],
      pending: [
        riskyPending(),
        riskyPending({ subject: "ls", resources: ["ls"] }),
      ],
      events: [
        { data },
        { data: { sessionID: ROOT, requestID: "perm_2", reply: "once" } },
      ],
    })

    const entries = ctx.approvalHistory.recent(ROOT, 10)
    expect(entries.length).toBe(1)
    expect(entries[0]?.subject).toBe("ls")
  })

  it("ignores unrelated event types", async () => {
    const ctx = await driveReplied({
      requests: [STATUS_REQUEST],
      pending: [riskyPending()],
      events: [
        {
          type: "session.idle",
          data: { sessionID: ROOT, requestID: "perm_1", reply: "once" },
        },
        {
          type: "permission.asked",
          data: { sessionID: ROOT, requestID: "perm_1", reply: "once" },
        },
      ],
    })

    expect(ctx.approvalHistory.recent(ROOT, 10)).toEqual([])
    // Untouched: the replied path never ran for those types.
    expect(ctx.pendingSubjects.take(STATUS_KEY)).not.toBeNull()
  })
})
