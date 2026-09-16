import type { DelegatedAccessConfig } from "../config.ts"
import {
  extractLastUserMessages,
  extractLatestAssistantModel,
  getSessionMessages,
} from "../ui/messages.ts"
import {
  classifyCommand,
  classifyDirectory,
  type ClassifyFailureClass,
  type ClassifierSession,
} from "../classifier/classify.ts"
import {
  runFailureNotificationInBackground,
  FailureNotifyRateLimiter,
} from "./failure-notify.ts"
import { resolveClassifierModel, type ModelRef } from "../classifier/model.ts"
import { resolveRootSessionID } from "../ui/session-tree.ts"
import { DirectoryVerdictCache } from "./directory-cache.ts"
import { ApprovalHistoryStore } from "./approval-history.ts"
import { PendingSubjectsMap } from "./pending-subjects.ts"
import { runSafePath } from "./safe-path.ts"
import type { SafePathBatcher } from "./safe-path-batcher.ts"
import { runRiskyPathInBackground } from "./risky-path.ts"
import type { Logger } from "../log.ts"
import {
  isDualRepoContext,
  type RepoContext,
  type DualRepoContext,
} from "../repo-context.ts"

/**
 * Structural slices of the V2 plugin domains this handler needs. (Structural
 * typing keeps us resilient to minor shape drift — the V2 package root
 * doesn't export domain types directly.)
 */
export type SessionAccess = ClassifierSession & {
  get(input: { sessionID: string }): Promise<{ parentID?: string } | undefined>
  context(input: {
    sessionID: string
  }): Promise<
    Array<{
      type?: string
      text?: string
      model?: { providerID?: string; id?: string }
    }> | undefined
  >
}
export type PermissionAccess = {
  list(input: { sessionID: string }): Promise<
    Array<{
      id: string
      sessionID: string
      action: string
      resources: Array<string>
    }> | undefined
  >
  reply(input: {
    sessionID: string
    requestID: string
    reply: "once" | "always" | "reject"
  }): Promise<unknown>
}
export type OpencodeAccess = {
  session: SessionAccess
  permission: PermissionAccess
}

/**
 * The V2 permission evaluate-hook event. `effect` is mutable: set it to
 * `"allow"` to auto-approve BEFORE opencode shows its TUI prompt; leave it
 * as `"ask"` to fall through to the normal prompt.
 */
export type PermissionEvaluation = {
  sessionID: string
  action: string
  resources: ReadonlyArray<string>
  effect: "allow" | "deny" | "ask"
  message?: string
}

/**
 * Permission actions that our plugin classifies for shell commands.
 *
 * V2 renamed the bash action to `shell`; older names are matched
 * defensively. The `action` is `string` (no enum), so we match loosely.
 */
const BASH_ACTION_MATCHES = new Set(["shell", "bash", "command"])

/** Runtime permission action for external-directory access. */
const EXTERNAL_DIRECTORY_ACTION = "external_directory"

export type HandlerContext = {
  opencode: OpencodeAccess
  config: DelegatedAccessConfig
  /**
   * The session's currently-configured model, used to pick a small default
   * classifier model when `config.classifierModel` is not set. `undefined` is
   * allowed (we just fall back to config-override / latest-assistant-model).
   */
  sessionModel: ModelRef | undefined
  /**
   * Track IDs of ephemeral classifier sessions we create. Used by the plugin
   * entry as a loop-guard: if a permission evaluation's sessionID is in this
   * set, the plugin skips it (defense-in-depth — the classifier's tools are
   * cleared in the session context hook and shouldn't request permissions,
   * but we guard anyway).
   */
  ephemeralSessionIDs: Set<string>
  /**
   * Registry mapping an ephemeral classifier session ID to the system prompt
   * it should use, read by the `session.hook("context")` handler in
   * src/index.ts to REPLACE opencode's global system preamble/instructions
   * with the classifier prompt (otherwise the classifier inherits e.g. "you
   * MUST invoke the using-superpowers skill" and never emits a VERDICT).
   */
  ephemeralSystemRegistry: import("../classifier/ephemeral-system.ts").EphemeralSystemRegistry
  /**
   * Shared TTL cache for recent SAFE external_directory verdicts. A single
   * instance is held for the plugin's lifetime and shared across all
   * permission events so burst requests for the same path skip the LLM call.
   */
  directoryVerdictCache: DirectoryVerdictCache
  /**
   * Per-plugin-lifetime store of recent human approval/rejection decisions
   * scoped by root session ID. Read by the handler before classification
   * (to surface priors to the classifier) and written by the
   * `permission.replied` event handler in `index.ts` when the human
   * actually resolves a permission.
   */
  approvalHistory: ApprovalHistoryStore
  /**
   * Short-lived map of `permissionKey → { rootSessionID, subject, ... }`
   * populated when a permission evaluation fires and drained when the
   * matching `permission.replied` event arrives. V2's evaluate event carries
   * no permission ID, so entries are keyed by
   * `sessionID + "\n" + action + "\n" + resources.join("\n")` (see
   * {@link evaluationKey}); the replied event carries the real requestID,
   * which we resolve to a key via the permission list.
   */
  pendingSubjects: PendingSubjectsMap
  /**
   * Shared batcher for SAFE-path notifications. Coalesces concurrent
   * notifications (e.g. burst external_directory requests) into a single
   * desktop notification so they don't cancel each other out.
   */
  safePathBatcher: SafePathBatcher
  /**
   * Shared, plugin-lifetime rate limiter for classifier-failure
   * notifications. Collapses a burst of failures (e.g. during a provider
   * outage) into a single notification per cooldown window.
   */
  failureNotifyRateLimiter: FailureNotifyRateLimiter
  /** Logger for diagnostic output. */
  log: Logger
  /**
   * Lazy fetcher for repo context (pinned + live), shared across
   * permission events. Returns `null` when the worktree isn't a git repo
   * or fetching fails. Keep it on `ctx` so the handler stays decoupled
   * from the cache implementation.
   *
   * Production callers return a {@link DualRepoContext} so the classifier
   * can detect pinned-vs-current mismatch. The legacy single-snapshot
   * `RepoContext` return type is still accepted so existing tests don't
   * have to be rewritten.
   */
  getRepoContext?: () => Promise<DualRepoContext | RepoContext | null>
}

/**
 * Stable key for a permission evaluation. Used because the V2 evaluate event
 * carries no permission ID; see {@link HandlerContext.pendingSubjects}.
 */
export function evaluationKey(ev: {
  sessionID: string
  action: string
  resources: ReadonlyArray<string>
}): string {
  return [ev.sessionID, ev.action, ...ev.resources].join("\n")
}

/**
 * React to a permission evaluation from opencode's V2 permission
 * evaluate hook.
 *
 * The hook fires BEFORE the TUI prompt is created. Setting
 * `ev.effect = "allow"` auto-approves with no flash; leaving `effect`
 * untouched (or any failure path) falls through to the normal prompt.
 * Everything in here is fail-closed: on any uncertainty we return without
 * touching `ev.effect` and the human decides.
 */
export async function handlePermissionEvent(
  ev: PermissionEvaluation,
  ctx: HandlerContext,
): Promise<void> {
  const { log } = ctx

  const base = {
    permissionAction: ev.action,
    permissionSessionID: ev.sessionID,
  }

  // Disabled → let opencode's normal approval machinery handle it.
  if (!ctx.config.enabled) {
    log.info("skip: plugin disabled", base)
    return
  }

  // Dispatch by permission action.
  if (BASH_ACTION_MATCHES.has(ev.action)) {
    const command = extractBashCommand(ev.resources)
    if (command === null) {
      log.info("skip: no command in resources", {
        ...base,
        resources: ev.resources as unknown,
      })
      return
    }
    await handleSubjectPermission({
      subject: command,
      subjectLabel: "command",
      directory: false,
      ev,
      ctx,
      base,
    })
    return
  }

  if (ev.action === EXTERNAL_DIRECTORY_ACTION) {
    if (!ctx.config.externalDirectoryEnabled) {
      log.info("skip: external_directory auto-approval disabled", base)
      return
    }
    const path = extractFirstResource(ev.resources)
    if (path === null) {
      log.info("skip: no path in external_directory resources", {
        ...base,
        resources: ev.resources as unknown,
      })
      return
    }
    await handleSubjectPermission({
      subject: path,
      subjectLabel: "path",
      directory: true,
      ev,
      ctx,
      base,
    })
    return
  }

  log.info("skip: unsupported permission action", base)
}

// ---------------------------------------------------------------------------
// Shared core: classify a subject, run safe/risky path, respond to permission
// ---------------------------------------------------------------------------

/**
 * Shared classification + response flow for any permission subject (shell
 * command or directory path). The two permission actions differ only in:
 *   - `subject` string (the thing being classified)
 *   - `directory` (selects the bash vs directory classifier + the burst
 *     deduplication cache lookup)
 */
async function handleSubjectPermission(args: {
  subject: string
  subjectLabel: "command" | "path"
  directory: boolean
  ev: PermissionEvaluation
  ctx: HandlerContext
  base: Record<string, unknown>
}): Promise<void> {
  const { subject, subjectLabel, directory, ev, ctx, base } = args
  const { log } = ctx

  // ---- Root-session resolution -------------------------------------------
  //
  // When a permission fires inside a subagent session, the sessionID points
  // at the subagent — whose "user" messages are the dispatching agent's
  // prompts, NOT the real human's. Walk up the parentID chain to the root.
  //
  // Fail-closed: null → TUI prompt remains, user decides manually.
  const rootSessionID = await resolveRootSessionID(
    ctx.opencode.session,
    ev.sessionID,
  )
  if (rootSessionID === null) {
    log.warn(
      "skip: could not resolve root session (fail-closed to TUI prompt)",
      base,
    )
    return
  }
  if (rootSessionID !== ev.sessionID) {
    log.info("resolved subagent to root session", {
      ...base,
      rootSessionID,
    })
  }

  // ---- Seed pending-subject map ------------------------------------------
  //
  // We seed BEFORE the directory-cache lookup so that on a cache hit (which
  // skips the classifier) the replied-event handler can still match the
  // permission back to its subject text. The classifier-verdict fields
  // are filled in later (after classification) and the autoApproved flag
  // is set inside `runSafeOrRiskyPath` when we resolve a SAFE verdict.
  const key = evaluationKey(ev)
  ctx.pendingSubjects.set(key, {
    rootSessionID,
    subject,
    subjectLabel,
    classifierVerdict: null,
    classifierReason: null,
    autoApproved: false,
  })

  // ---- Directory cache lookup (directories only) -------------------------
  if (directory) {
    const cacheKey = DirectoryVerdictCache.keyFor([...ev.resources])
    const cached = ctx.directoryVerdictCache.get(cacheKey)
    if (cached) {
      log.info("directory cache hit — skipping classifier", {
        ...base,
        [subjectLabel]: subject,
        cachedVerdict: cached.verdict.verdict,
        cachedReason: cached.verdict.reason,
      })
      // Update the pending entry with the cached verdict so the replied
      // handler has a verdict to record (even on the cache-hit path).
      ctx.pendingSubjects.update(key, (cur) => ({
        ...cur,
        classifierVerdict: cached.verdict.verdict,
        classifierReason: cached.verdict.reason,
      }))
      // Run safe-path with the cached verdict (burst requests still get the
      // countdown; user can cancel any of them).
      await runSafeOrRiskyPath({
        verdict: cached.verdict,
        subject,
        subjectLabel,
        ev,
        ctx,
        base,
      })
      return
    }
  }

  // ---- Message extraction ------------------------------------------------
  let entries
  try {
    entries = await getSessionMessages(ctx.opencode.session, rootSessionID)
  } catch (e) {
    log.error("getSessionMessages failed", {
      ...base,
      error: e instanceof Error ? e.message : String(e),
    })
    return
  }

  const userMessages = extractLastUserMessages(
    entries,
    ctx.config.contextMessageCount,
  )
  const fallbackModel = extractLatestAssistantModel(entries)

  // ---- Classifier model --------------------------------------------------
  const model = resolveClassifierModel({
    configOverride: ctx.config.classifierModel,
    sessionModel: ctx.sessionModel ?? fallbackModel ?? undefined,
  })
  if (!model) {
    log.warn("skip: no classifier model could be resolved", {
      ...base,
      hasCtxSessionModel: Boolean(ctx.sessionModel),
      hasFallbackModel: Boolean(fallbackModel),
      hasConfigOverride: Boolean(ctx.config.classifierModel),
    })
    return
  }

  const modelSource = ctx.config.classifierModel
    ? "configOverride"
    : ctx.sessionModel
      ? "ctxSessionModel"
      : fallbackModel
        ? "latestAssistantMessage"
        : "unknown"

  // ---- Prior approvals ---------------------------------------------------
  //
  // Read recent in-session human decisions and surface them to the
  // classifier as prior-decision evidence. When disabled by config, pass
  // an empty array so the classifier prompt omits the block entirely.
  const priorApprovals = ctx.config.approvalHistoryEnabled
    ? ctx.approvalHistory.recent(rootSessionID, ctx.config.approvalHistoryMax)
    : []

  // Best-effort fetch of repo context (branch + open PR). Cached upstream
  // with a short TTL so back-to-back permissions don't all re-fetch.
  let repoContext: DualRepoContext | RepoContext | null = null
  if (ctx.getRepoContext) {
    try {
      repoContext = await ctx.getRepoContext()
    } catch (e) {
      // Repo context is optional — never let a fetch failure block
      // classification. Log so the failure mode is debuggable; downstream
      // continues with `null` repo context.
      log.warn("getRepoContext threw; continuing without repo context", {
        ...base,
        error: e instanceof Error ? e.message : String(e),
      })
      repoContext = null
    }
  }

  log.info("classifying", {
    ...base,
    [subjectLabel]: subject,
    classifierModel: `${model.providerID}/${model.modelID}`,
    modelSource,
    sessionBranch: pickBranch(repoContext, "pinned"),
    sessionOpenPR: pickOpenPR(repoContext, "pinned"),
    currentBranch: pickBranch(repoContext, "current"),
    currentOpenPR: pickOpenPR(repoContext, "current"),
    priorApprovalCount: priorApprovals.length,
  })

  // ---- Classifier call ---------------------------------------------------
  const commonClassifyArgs = {
    session: ctx.opencode.session,
    userMessages,
    parentSessionID: ev.sessionID,
    model,
    timeoutMs: ctx.config.classifierTimeoutMs,
    repoContext,
    priorApprovals,
    log,
    retries: ctx.config.classifierRetries,
    onEphemeralSessionCreated: (id: string, systemPrompt: string) => {
      ctx.ephemeralSessionIDs.add(id)
      ctx.ephemeralSystemRegistry.set(id, systemPrompt)
    },
    onEphemeralSessionDeleted: (id: string) => {
      ctx.ephemeralSessionIDs.delete(id)
      ctx.ephemeralSystemRegistry.delete(id)
    },
  }

  // Capture the FINAL failure class reported by the classifier (after any
  // retries) so the failure-notification path can distinguish a transient
  // timeout from a harder error. Defaults to "error" if the classifier
  // returns null without reporting (shouldn't happen, but fail safe).
  let failureClass: ClassifyFailureClass = "error"
  const onFailure = (fc: ClassifyFailureClass) => {
    failureClass = fc
  }

  const verdict = directory
    ? // Directory path: directory-specific classifier prompt.
      await classifyDirectory({ ...commonClassifyArgs, path: subject, onFailure })
    : // Shell path: use the convenience wrapper that supplies the bash prompt.
      await classifyCommand({ ...commonClassifyArgs, command: subject, onFailure })

  if (!verdict) {
    log.warn("classifier failed; leaving TUI prompt alone", {
      ...base,
      failureClass,
    })
    maybeNotifyClassifierFailure({
      ctx,
      ev,
      subject,
      failureClass,
      base,
    })
    return
  }

  log.info("classifier verdict", {
    ...base,
    verdict: verdict.verdict,
    reason: verdict.reason,
  })

  // ---- Update pending subject with the verdict ---------------------------
  ctx.pendingSubjects.update(key, (cur) => ({
    ...cur,
    classifierVerdict: verdict.verdict,
    classifierReason: verdict.reason,
  }))

  // ---- Directory cache population (SAFE only) ----------------------------
  if (directory && verdict.verdict === "SAFE") {
    const cacheKey = DirectoryVerdictCache.keyFor([...ev.resources])
    ctx.directoryVerdictCache.set(
      cacheKey,
      verdict,
      ctx.config.directoryVerdictCacheTtlMs,
    )
  }

  // ---- Safe / Risky path -------------------------------------------------
  await runSafeOrRiskyPath({
    verdict,
    subject,
    subjectLabel,
    ev,
    ctx,
    base,
  })
}

// ---------------------------------------------------------------------------
// Classifier-failure notification (rate-limited, Reject-only)
// ---------------------------------------------------------------------------

/**
 * On a classifier failure (after retries), optionally fire a rate-limited,
 * informational + Reject-only desktop notification so the human gets insight
 * that a transient error happened instead of a silent fall-through to the TUI
 * prompt. Gated by `config.notifyOnClassifierFailure` and the shared
 * `failureNotifyRateLimiter`. Fire-and-forget — never blocks the handler.
 */
function maybeNotifyClassifierFailure(args: {
  ctx: HandlerContext
  ev: PermissionEvaluation
  subject: string
  failureClass: ClassifyFailureClass
  base: Record<string, unknown>
}): void {
  const { ctx, ev, subject, failureClass, base } = args
  if (!ctx.config.notifyOnClassifierFailure) return

  const decision = ctx.failureNotifyRateLimiter.register(
    failureClass,
    Date.now(),
  )
  if (!decision.notify) {
    ctx.log.debug("classifier-failure notification suppressed (rate-limited)", {
      ...base,
      failureClass,
    })
    return
  }

  ctx.log.info("firing classifier-failure notification", {
    ...base,
    failureClass,
    suppressedCount: decision.suppressedCount,
  })

  void runFailureNotificationInBackground({
    reply: makeReplier(ctx, ev),
    command: subject,
    failureClass,
    suppressedCount: decision.suppressedCount,
    sound: ctx.config.notificationSound,
    timeoutSec: 60,
    log: ctx.log,
  })
}

// ---------------------------------------------------------------------------
// Safe / risky path execution (shared between cache-hit and fresh-verdict paths)
// ---------------------------------------------------------------------------

async function runSafeOrRiskyPath(args: {
  verdict: import("../classifier/parse.ts").Verdict
  subject: string
  subjectLabel: "command" | "path"
  ev: PermissionEvaluation
  ctx: HandlerContext
  base: Record<string, unknown>
}): Promise<void> {
  const { verdict, subject, subjectLabel, ev, ctx, base } = args
  const { log } = ctx

  if (verdict.verdict === "SAFE") {
    log.info("entering safe-path", {
      ...base,
      countdownMs: ctx.config.safeCountdownMs,
    })
    const decision = await runSafePath({
      command: subject,
      reason: verdict.reason,
      countdownMs: ctx.config.safeCountdownMs,
      sound: ctx.config.notificationSound,
      log,
      batcher: ctx.safePathBatcher,
    })
    log.info("safe-path returned", { ...base, decision })
    if (decision === "allow") {
      log.info("auto-approving", {
        ...base,
        [subjectLabel]: subject,
      })
      // Tag the pending entry BEFORE setting effect so that even if the
      // server emits `permission.replied` immediately after, the replied-
      // event handler sees `autoApproved: true` and filters this out of the
      // human-decision history.
      ctx.pendingSubjects.update(evaluationKey(ev), (cur) => ({
        ...cur,
        autoApproved: true,
      }))
      // Auto-approve BEFORE the TUI prompt exists — the V2 evaluate hook's
      // whole point.
      ev.effect = "allow"
    } else {
      log.info("user cancelled auto-approval; TUI prompt remains", base)
    }
    return
  }

  log.info("risky — escalating via TUI + notification", base)
  // RISKY: fire the notification alongside opencode's TUI prompt.
  void runRiskyPathInBackground({
    reply: makeReplier(ctx, ev),
    command: subject,
    reason: verdict.reason,
    sound: ctx.config.notificationSound,
    timeoutSec: 60,
    log: ctx.log,
  })
}

// ---------------------------------------------------------------------------
// Programmatic permission resolution (notification-button replies)
// ---------------------------------------------------------------------------

/**
 * Build a replier that resolves the pending permission request for `ev` at
 * reply time and answers it.
 *
 * Why lazy: the evaluate hook fires BEFORE opencode has assigned the
 * permission a request ID (the request may not even exist yet when the
 * classification fails). Notification buttons are clicked seconds later, by
 * which time the request is registered — so we look it up then, matching on
 * sessionID + action + resources. A short retry covers the race where the
 * request lands between our hook returning and the user clicking.
 */
function makeReplier(
  ctx: HandlerContext,
  ev: PermissionEvaluation,
): (response: "once" | "always" | "reject") => Promise<void> {
  return async (response) => {
    const { log } = ctx

    let requestID: string | null = null
    // A few quick attempts: immediately, then two short backoffs. All
    // best-effort — total added latency is bounded (~450ms) and only paid
    // when the user actually clicked a notification button.
    for (let attempt = 0; attempt < 3 && requestID === null; attempt++) {
      if (attempt > 0) await sleep(200)
      try {
        const requests = await ctx.opencode.permission.list({
          sessionID: ev.sessionID,
        })
        requestID =
          requests?.find(
            (r) =>
              r.sessionID === ev.sessionID &&
              r.action === ev.action &&
              sameResources(r.resources, ev.resources),
          )?.id ?? null
      } catch (e) {
        log.warn("permission list failed while resolving request", {
          error: e instanceof Error ? e.message : String(e),
        })
      }
    }

    if (requestID === null) {
      // Fail-closed: don't guess. The TUI prompt remains for the user.
      log.warn("could not resolve permission requestID; not replying", {
        permissionAction: ev.action,
        response,
      })
      return
    }

    await ctx.opencode.permission.reply({
      sessionID: ev.sessionID,
      requestID,
      reply: response,
    })
    log.info("permission reply succeeded", {
      permissionAction: ev.action,
      requestID,
      response,
    })
  }
}

function sameResources(
  a: ReadonlyArray<string>,
  b: ReadonlyArray<string>,
): boolean {
  if (a.length !== b.length) return false
  return a.every((v, i) => v === b[i])
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// ---------------------------------------------------------------------------
// Resource extraction helpers
// ---------------------------------------------------------------------------

/**
 * Coerce the resources array into a single string, taking only the FIRST
 * element. Returns `null` when no usable value is present.
 *
 * Used for the external_directory action, where the array holds independent
 * paths (a burst), not segments of one compound command — there the first
 * element is the representative display subject and the full list drives the
 * cache key separately.
 */
function extractFirstResource(
  resources: ReadonlyArray<string>,
): string | null {
  const first = resources[0]
  if (typeof first === "string" && first.length > 0) return first
  return null
}

/**
 * Coerce the shell resources array into the FULL command to classify.
 *
 * The scanner splits a compound shell command (sub-commands joined by `&&`,
 * `;`, `|`, etc.) into its constituent pieces and delivers them as the
 * resources array — e.g. `git add . && git commit -m x` arrives as
 * `["git add .", "git commit -m x"]`. Classifying only the first element
 * judged a different, frequently safer command than what actually runs,
 * letting a benign leading segment mask a risky trailing one.
 *
 * We therefore re-join all non-empty segments with ` && ` so the classifier
 * sees the entire command. A single-element array is returned unchanged.
 * Returns `null` when there is no usable command text.
 */
function extractBashCommand(
  resources: ReadonlyArray<string>,
): string | null {
  const segments = resources.filter(
    (p): p is string => typeof p === "string" && p.length > 0,
  )
  if (segments.length === 0) return null
  return segments.join(" && ")
}

/**
 * Extract a branch name from either a single or dual repo context for
 * structured logging. Returns null when the requested side is unavailable.
 */
function pickBranch(
  repo: DualRepoContext | RepoContext | null,
  side: "pinned" | "current",
): string | null {
  if (!repo) return null
  if (isDualRepoContext(repo)) {
    return repo[side]?.branch ?? null
  }
  // Legacy single shape — log it under the "current" side only.
  return side === "current" ? repo.branch ?? null : null
}

/**
 * Extract an open-PR number from either a single or dual repo context for
 * structured logging. Returns null when the requested side has no open PR
 * or is unavailable.
 */
function pickOpenPR(
  repo: DualRepoContext | RepoContext | null,
  side: "pinned" | "current",
): string | null {
  if (!repo) return null
  if (isDualRepoContext(repo)) {
    return repo[side]?.openPR?.number !== undefined
      ? String(repo[side]?.openPR?.number)
      : null
  }
  return side === "current" ? (repo.openPR?.number !== undefined ? String(repo.openPR.number) : null) : null
}
