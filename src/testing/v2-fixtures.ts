/**
 * Shared V2 test fixtures.
 *
 * Why this exists: under V1 each suite rebuilt its own mocks for the plugin
 * input, client, session, and message shapes. When the plugin moved to the V2
 * API those copies drifted apart, and the V2 port left ~160 tests failing on
 * mock shapes rather than on behavior. One builder per boundary keeps that
 * from recurring.
 *
 * Everything here is a plain object or a `vi.fn()`. Nothing touches the
 * network, the clock, or the real opencode domains.
 */
import { vi } from "vitest"
import type { Logger } from "../log.ts"
import type {
  HandlerContext,
  OpencodeAccess,
  PermissionEvaluation,
} from "../permission/handler.ts"
import { DirectoryVerdictCache } from "../permission/directory-cache.ts"
import { SafePathBatcher } from "../permission/safe-path-batcher.ts"
import { ApprovalHistoryStore } from "../permission/approval-history.ts"
import { PendingSubjectsMap } from "../permission/pending-subjects.ts"
import { FailureNotifyRateLimiter } from "../permission/failure-notify.ts"
import { EphemeralSystemRegistry } from "../classifier/ephemeral-system.ts"
import { parseConfig } from "../config.ts"
import type { NotifyActionResult } from "../notify/notify.ts"
import type { TranscriptMessage } from "../ui/messages.ts"
import type { ModelRef } from "../classifier/model.ts"

/** A logger whose output is captured instead of written. */
export function makeLogger() {
  return {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  } satisfies Logger
}

/**
 * The V2 `ctx.session` slice. `create`/`generate`/`interrupt`/`get`/`context`
 * default to the happy path; individual tests override with
 * `mockResolvedValueOnce`.
 */
export function makeSessionDomain() {
  return {
    create: vi.fn(async (_input?: unknown) => ({ id: "sess_ephemeral" })),
    generate: vi.fn(async (_input?: unknown) => ({
      text: "VERDICT: SAFE\nREASON: fixture default",
    })),
    interrupt: vi.fn(async (_input?: unknown) => ({})),
    get: vi.fn(async (_input?: unknown) => undefined),
    context: vi.fn(async (_input?: unknown) => [] as TranscriptMessage[]),
    hook: vi.fn(async (_name: string, _cb: unknown) => ({
      dispose: async () => {},
    })),
  }
}

/** The V2 `ctx.permission` slice. */
export function makePermissionDomain() {
  return {
    list: vi.fn(async (_input?: unknown) => []),
    reply: vi.fn(async (_input?: unknown) => ({})),
    hook: vi.fn(async (_name: string, _cb: unknown) => ({
      dispose: async () => {},
    })),
  }
}

/**
 * A push-driven `ctx.event.subscribe()` stream. Tests `push()` an event and
 * the plugin's background `for await` loop sees it.
 */
export function makeEventStream() {
  const queue: unknown[] = []
  const waiters: Array<(value: IteratorResult<unknown>) => void> = []
  let closed = false

  const stream = {
    [Symbol.asyncIterator]() {
      return {
        next(): Promise<IteratorResult<unknown>> {
          const queued = queue.shift()
          if (queued !== undefined) {
            return Promise.resolve({ value: queued, done: false })
          }
          if (closed) return Promise.resolve({ value: undefined, done: true })
          return new Promise((resolve) => waiters.push(resolve))
        },
      }
    },
  }

  return {
    /** `ctx.event.subscribe()` — resolves to the async iterable. */
    subscribe: vi.fn(async () => stream),
    push(event: unknown) {
      const waiter = waiters.shift()
      if (waiter) waiter({ value: event, done: false })
      else queue.push(event)
    },
    close() {
      closed = true
      let waiter = waiters.shift()
      while (waiter) {
        waiter({ value: undefined, done: true })
        waiter = waiters.shift()
      }
    },
  }
}

/**
 * The V2 plugin `Context` passed to `Plugin.setup`. Only the domains the
 * plugin actually touches are populated.
 */
export function makePluginContext(
  overrides: { directory?: string; options?: unknown } = {},
) {
  const session = makeSessionDomain()
  const permission = makePermissionDomain()
  const event = makeEventStream()

  return {
    /** Cast at the call site: `DelegatedAccess.setup(ctx as never)`. */
    ctx: {
      location: { directory: overrides.directory ?? "/tmp/repo" },
      options: overrides.options,
      session,
      permission,
      event,
      agent: {},
      shell: {},
    },
    session,
    permission,
    event,
  }
}

/**
 * The `HandlerContext` the permission handler runs against. Stores are real
 * instances so tests exercise their actual semantics rather than a mock's.
 */
export function makeHandlerContext(
  overrides: Partial<HandlerContext> = {},
): HandlerContext {
  const session = makeSessionDomain()
  const permission = makePermissionDomain()

  const base: HandlerContext = {
    opencode: { session, permission } as unknown as OpencodeAccess,
    config: parseConfig(undefined),
    sessionModel: undefined,
    ephemeralSessionIDs: new Set<string>(),
    ephemeralSystemRegistry: new EphemeralSystemRegistry(),
    directoryVerdictCache: new DirectoryVerdictCache(),
    approvalHistory: new ApprovalHistoryStore({ maxPerSession: 10 }),
    pendingSubjects: new PendingSubjectsMap(),
    safePathBatcher: new SafePathBatcher({
      batchWindowMs: 0,
      sendNotification: async () => ({ type: "timeout" }) as NotifyActionResult,
      countdownMs: 0,
      sound: false,
      log: makeLogger(),
    }),
    failureNotifyRateLimiter: new FailureNotifyRateLimiter({ cooldownMs: 0 }),
    log: makeLogger(),
    getRepoContext: async () => ({ pinned: null, current: null }),
  }

  return { ...base, ...overrides }
}

/** A V2 permission evaluate event. `effect` starts as "ask" (fail-closed). */
export function makeEvaluation(
  overrides: Partial<PermissionEvaluation> = {},
): PermissionEvaluation {
  return {
    sessionID: "sess_root",
    action: "shell",
    resources: ["git status"],
    effect: "ask",
    ...overrides,
  }
}

/** A V2 transcript user message. */
export function userMessage(text: string): TranscriptMessage {
  return { type: "user", text }
}

/** A V2 transcript assistant message, optionally with a model ref. */
export function assistantMessage(
  model?: { providerID: string; id: string },
  text = "",
): TranscriptMessage {
  return { type: "assistant", text, ...(model ? { model } : {}) }
}

/** A `PermissionReplier` spy, as injected by the handler. */
export function makeReply(impl?: () => Promise<void>) {
  return vi.fn(impl ?? (async () => {}))
}

/** A `ModelRef` for classifier-model tests. */
export const SAMPLE_MODEL: ModelRef = {
  providerID: "anthropic",
  modelID: "claude-haiku-4",
}
