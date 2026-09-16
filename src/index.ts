import { execFile } from "child_process"
import { Plugin } from "@opencode/plugin"
import { parseConfig, type DelegatedAccessConfig } from "./config.ts"
import {
  handlePermissionEvent,
  evaluationKey,
  type HandlerContext,
  type OpencodeAccess,
} from "./permission/handler.ts"
import { DirectoryVerdictCache } from "./permission/directory-cache.ts"
import { SafePathBatcher } from "./permission/safe-path-batcher.ts"
import { ApprovalHistoryStore } from "./permission/approval-history.ts"
import { PendingSubjectsMap } from "./permission/pending-subjects.ts"
import { FailureNotifyRateLimiter } from "./permission/failure-notify.ts"
import {
  EphemeralSystemRegistry,
  registerEphemeralIsolationHooks,
} from "./classifier/ephemeral-system.ts"
import { sendNotification } from "./notify/notify.ts"
import type { ModelRef } from "./classifier/model.ts"
import { createLogger, type Logger } from "./log.ts"
import {
  RepoContextCache,
  type BunShellLike,
  type DualRepoContext,
} from "./repo-context.ts"
import { SessionRepoContext } from "./session-repo-context.ts"

/**
 * OpenCode V2 plugin entry point.
 *
 * Hook wiring (V2 of the plugin API):
 *
 *   1. `permission.hook("evaluate")` — fires BEFORE the TUI prompt exists.
 *      We classify the subject; on SAFE (after the countdown) we set
 *      `ev.effect = "allow"` to auto-approve with no flash. Everything is
 *      fail-closed: any error or uncertainty leaves `effect` untouched and
 *      the human decides in the TUI.
 *   2. `session.hook("context")` — for OUR ephemeral classifier sessions
 *      only, replace the assembled `system` array with just the registered
 *      classifier prompt and clear `tools` entirely. (V1 used the
 *      `experimental.chat.system.transform` hook + a per-prompt tools map;
 *      the context hook replaces both and is stronger — a user allowlist
 *      can't re-enable tools for the classifier.)
 *   3. `event.subscribe()` — consumed in the background for
 *      `permission.replied`, feeding the approval-history store.
 *
 * All diagnostic output goes through the console-backed logger; opencode
 * captures plugin stdout/stderr into its log file. Grep with:
 *
 *     grep delegated-access ~/.local/share/opencode/log/*.log
 *
 * Plugin config: per-plugin tuple options in opencode.json (parsed by
 * parseConfig; invalid shapes fall back to defaults silently rather than
 * crashing opencode).
 */
const DelegatedAccess = Plugin.define({
  id: "opencode-delegated-access",
  async setup(ctx) {
    const directory = ctx.location.directory
    const log: Logger = createLogger()
    log.info("plugin loaded")

    // V2 exposes no Bun `$`; repo-context shells out through a tiny
    // execFile shim (argv-slot safe, non-zero exit → non-zero exitCode).
    const shell: BunShellLike = (cmd, cwd) =>
      new Promise((resolve) => {
        execFile(
          cmd[0],
          cmd.slice(1),
          { cwd, timeout: 15_000 },
          (err, stdout) => {
            if (err && (err as NodeJS.ErrnoException).code === "ENOENT") {
              resolve(null) // binary missing (e.g. gh not installed)
              return
            }
            resolve({
              exitCode: err && typeof (err as { code?: unknown }).code === "number"
                ? ((err as { code: number }).code as number)
                : err
                  ? 1
                  : 0,
              stdout: typeof stdout === "string" ? stdout : "",
            })
          },
        )
      })

    // Live repo-context cache: branch + open PR (via gh) — refreshed on a
    // short TTL so the classifier always has an up-to-date view of where
    // the agent thinks it is. Keyed by cwd (in practice always `directory`).
    const repoContextCache = new RepoContextCache({
      $: shell,
    })

    // Session-pinned repo context: captured exactly once (lazily on the
    // first permission event) and frozen for the lifetime of the plugin
    // process. Compared against the live `repoContextCache` so the
    // classifier can detect when the agent has moved off the human's
    // pre-committed branch/PR.
    // Share the live cache's fetcher so the first permission event only
    // pays for ONE git+gh round-trip — the session pin reads through the
    // cache's first-call result and freezes it forever, while the cache
    // continues refreshing it on its normal TTL.
    const sessionRepoContext = new SessionRepoContext({
      worktree: directory,
      fetcher: (cwd) => repoContextCache.get(cwd),
    })

    // Track whether we've already logged a "repo context unavailable" line
    // so we don't spam the log on every permission event when gh is missing.
    let loggedRepoContextUnavailable = false

    async function getRepoContext(): Promise<DualRepoContext> {
      const [pinned, current] = await Promise.all([
        sessionRepoContext.getPinned(),
        repoContextCache.get(directory),
      ])
      if (
        pinned === null &&
        current === null &&
        !loggedRepoContextUnavailable
      ) {
        log.info("repo context unavailable", { worktree: directory })
        loggedRepoContextUnavailable = true
      }
      return { pinned, current }
    }

    // Config is resolved at setup time from the per-plugin tuple options.
    let config: DelegatedAccessConfig
    try {
      config = parseConfig(ctx.options)
      log.info("config resolved", {
        source: ctx.options !== undefined ? "tuple" : "defaults",
        enabled: config.enabled,
        contextMessageCount: config.contextMessageCount,
        safeCountdownMs: config.safeCountdownMs,
        classifierTimeoutMs: config.classifierTimeoutMs,
        classifierRetries: config.classifierRetries,
        classifierModel: config.classifierModel,
        externalDirectoryEnabled: config.externalDirectoryEnabled,
        directoryVerdictCacheTtlMs: config.directoryVerdictCacheTtlMs,
        approvalHistoryEnabled: config.approvalHistoryEnabled,
        approvalHistoryMax: config.approvalHistoryMax,
        notifyOnClassifierFailure: config.notifyOnClassifierFailure,
        classifierFailureNotifyCooldownMs: config.classifierFailureNotifyCooldownMs,
      })
    } catch (e) {
      config = parseConfig(undefined)
      log.warn("invalid plugin options; using defaults", {
        error: e instanceof Error ? e.message : String(e),
      })
    }

    // Optional "model": "provider/model-id" tuple option used as the
    // classifier's fallback model before the latest-assistant-message
    // heuristic. (V1 latched the session's configured model from the config
    // hook; V2 exposes no equivalent, so the transcript heuristic covers it.)
    const sessionModel: ModelRef | undefined =
      parseModelString(
        (ctx.options as { model?: unknown } | undefined)?.model,
      ) ?? undefined

    // Track IDs of ephemeral classifier sessions we create. All permission
    // evaluations skip events whose `sessionID` is in this set, so the
    // classifier can't trigger itself (defense-in-depth — the context hook
    // also clears its tools entirely).
    const ephemeralSessionIDs = new Set<string>()

    // Maps each ephemeral classifier session ID to the system prompt it should
    // use. Read by the `session.hook("context")` handler below to REPLACE
    // opencode's global agent preamble/instructions for the classifier prompt
    // (otherwise the classifier inherits e.g. the superpowers "you MUST invoke
    // the skill" directive and replies conversationally instead of emitting a
    // VERDICT — observed as repeated parse failures in production).
    const ephemeralSystemRegistry = new EphemeralSystemRegistry()

    // Shared TTL cache for recent SAFE external_directory verdicts. Held at
    // plugin lifetime (not per-session) so burst deduplication works across
    // rapid-fire permission events on the same session.
    const directoryVerdictCache = new DirectoryVerdictCache()

    // Per-plugin-lifetime store of recent human approval/rejection decisions
    // scoped by root session ID. Surfaced to the classifier as prior-decision
    // evidence; written by the `permission.replied` event handler when the
    // human actually resolves a permission.
    const approvalHistory = new ApprovalHistoryStore({
      maxPerSession: config.approvalHistoryMax,
    })

    // Short-lived map of `evaluationKey → { rootSessionID, subject, ... }`
    // that bridges the gap between the rich subject info seen at evaluation
    // time and the `permission.replied` event (which carries only the
    // requestID — resolved back to a key via the permission list).
    const pendingSubjects = new PendingSubjectsMap()

    // Shared batcher for SAFE-path notifications. A single instance means all
    // concurrent permission events funnel through the same 200ms batch window,
    // so bursts (e.g. agent accessing 3 sub-directories at once) produce one
    // desktop notification instead of N notifications that cancel each other.
    const safePathBatcher = new SafePathBatcher({
      batchWindowMs: 200,
      sendNotification,
      countdownMs: config.safeCountdownMs,
      sound: config.notificationSound,
      log,
    })

    // Shared, plugin-lifetime rate limiter for classifier-failure
    // notifications. Held here (not per-session) so a burst of failures across
    // rapid permission events collapses into a single notification.
    const failureNotifyRateLimiter = new FailureNotifyRateLimiter({
      cooldownMs: config.classifierFailureNotifyCooldownMs,
    })

    // V2 domain slices handed to the handler. The branded-ID types on the
    // real domains don't structurally match plain strings, but the runtime
    // shapes are exactly these (verified against @opencode/client 2.0.3).
    const opencode = {
      session: ctx.session,
      permission: ctx.permission,
    } as unknown as OpencodeAccess

    function buildCtx(): HandlerContext {
      return {
        opencode,
        config,
        sessionModel,
        ephemeralSessionIDs,
        directoryVerdictCache,
        approvalHistory,
        pendingSubjects,
        safePathBatcher,
        failureNotifyRateLimiter,
        ephemeralSystemRegistry,
        log,
        getRepoContext,
      }
    }

    // --- Hook 1: permission evaluation (pre-prompt interception) ------------
    await ctx.permission.hook("evaluate", async (ev) => {
      // Loop-guard: skip evaluations from our own ephemeral classifier
      // sessions.
      if (ephemeralSessionIDs.has(ev.sessionID)) {
        log.debug("skip: ephemeral classifier session", {
          permissionAction: ev.action,
        })
        return
      }

      log.info("permission evaluate fired", {
        permissionAction: ev.action,
        resources: ev.resources as unknown,
        effect: ev.effect,
      })

      try {
        await handlePermissionEvent(ev, buildCtx())
      } catch (e) {
        // Fail-closed: an exception must never auto-approve OR break the
        // prompt. Leave `ev.effect` untouched and log.
        log.error("handler threw", {
          permissionAction: ev.action,
          error: e instanceof Error ? e.message : String(e),
        })
      }
    })

    // --- Hook 2: session context isolation for ephemeral classifier sessions
    // (system prompt replacement + total tool denial). Registered for ALL
    // sessions but a no-op for every session not in the ephemeral registry.
    await registerEphemeralIsolationHooks(
      ctx.session,
      ephemeralSessionIDs,
      ephemeralSystemRegistry,
      (sessionID) => log.debug("classifier context isolated", { sessionID }),
    )

    // --- Hook 3: permission.replied → approval history -----------------------
    const repliedEvents = await ctx.event.subscribe()
    void (async () => {
      for await (const event of repliedEvents) {
        const type = (event as { type?: unknown }).type
        if (type !== "permission.replied") continue

        const data = (event as { data?: unknown }).data
        const normalized = normalizeRepliedProperties(data)
        if (normalized === null) {
          log.warn("permission.replied: malformed event data", {
            data: data as unknown,
          })
          continue
        }

        // Resolve the request back to its action + resources so we can build
        // the pending-subject key. The request may already be gone (resolved)
        // — history capture is best-effort, same as V1.
        let action: string | null = null
        let resources: string[] = []
        try {
          const requests = await ctx.permission.list({
            sessionID: normalized.sessionID,
          })
          const req = requests?.find((r) => r.id === normalized.permissionID)
          if (req) {
            action = req.action
            resources = [...req.resources]
          }
        } catch (e) {
          log.warn("permission.replied: list failed", {
            error: e instanceof Error ? e.message : String(e),
          })
        }
        if (action === null) {
          log.debug(
            "permission.replied: request no longer listed; skipping history",
            { requestID: normalized.permissionID },
          )
          continue
        }

        handlePermissionReplied(
          {
            sessionID: normalized.sessionID,
            response: normalized.response,
            key: evaluationKey({
              sessionID: normalized.sessionID,
              action,
              resources,
            }),
          },
          {
            pendingSubjects,
            approvalHistory,
            config,
            log,
          },
        )
      }
    })().catch((e) => {
      log.error("replied-event loop crashed", {
        error: e instanceof Error ? e.message : String(e),
      })
    })
  },
})

/**
 * Pure handler for `permission.replied` events. Looks up the matching
 * pending subject (set by the evaluate path), filters out our own
 * auto-approvals, and appends a human-decision entry to the approval
 * history.
 */
function handlePermissionReplied(
  properties: {
    sessionID: string
    response: string
    key: string
  },
  deps: {
    pendingSubjects: PendingSubjectsMap
    approvalHistory: ApprovalHistoryStore
    config: DelegatedAccessConfig
    log: Logger
    now?: () => number
  },
): void {
  const { pendingSubjects, approvalHistory, config, log } = deps
  const now = deps.now ?? Date.now

  if (!config.approvalHistoryEnabled) {
    log.debug("permission.replied: history disabled", {
      key: properties.key,
    })
    return
  }

  const pending = pendingSubjects.take(properties.key)
  if (!pending) {
    log.debug("permission.replied: no pending subject for key", {
      key: properties.key,
    })
    return
  }

  if (pending.autoApproved) {
    log.debug("permission.replied: skipping our own auto-approval", {
      key: properties.key,
      subject: pending.subject,
    })
    return
  }

  const response = properties.response
  if (response !== "once" && response !== "always" && response !== "reject") {
    log.warn("permission.replied: unrecognised response value", {
      key: properties.key,
      response,
    })
    return
  }

  if (pending.classifierVerdict === null) {
    // Human resolved before classifier returned — still record, but with
    // a clear marker that we have no classifier verdict to associate.
    log.info("permission.replied: human resolved before classifier", {
      key: properties.key,
      response,
    })
  }

  approvalHistory.record(pending.rootSessionID, {
    subject: pending.subject,
    subjectLabel: pending.subjectLabel,
    response,
    classifierVerdict: pending.classifierVerdict ?? "RISKY",
    classifierReason:
      pending.classifierReason ??
      "(classifier did not complete before human resolved)",
    timestamp: now(),
  })

  log.info("recorded human approval decision", {
    rootSessionID: pending.rootSessionID,
    response,
    subject: pending.subject,
  })
}

/**
 * Normalise the data payload of a `permission.replied` event into
 * `{ sessionID, permissionID, response }`.
 *
 * V2 emits `{ sessionID, requestID, reply }` (verified against
 * @opencode/client 2.0.3). V1 emitted `{ sessionID, permissionID, response }`
 * with its own SDK/runtime drift. We accept both key sets — SDK-canonical
 * V2 names win when both are present — and return `null` for anything that
 * can't be coerced (the caller logs and skips).
 */
function normalizeRepliedProperties(
  raw: unknown,
): { sessionID: string; permissionID: string; response: string } | null {
  if (!raw || typeof raw !== "object") return null

  const r = raw as {
    sessionID?: unknown
    requestID?: unknown
    reply?: unknown
    permissionID?: unknown
    response?: unknown
  }

  if (typeof r.sessionID !== "string") return null

  const permissionID =
    typeof r.requestID === "string"
      ? r.requestID
      : typeof r.permissionID === "string"
        ? r.permissionID
        : null
  if (permissionID === null) return null

  const response =
    typeof r.reply === "string"
      ? r.reply
      : typeof r.response === "string"
        ? r.response
        : null
  if (response === null) return null

  return { sessionID: r.sessionID, permissionID, response }
}

/**
 * Parse a `model: "provider/model-id"` tuple option into a ModelRef, or
 * undefined if the input is missing/malformed. Model IDs may contain
 * slashes (e.g. openrouter's "anthropic/claude-haiku"), so we split on the
 * first slash only.
 */
function parseModelString(
  input: unknown,
): { providerID: string; modelID: string } | undefined {
  if (typeof input !== "string") return undefined
  const trimmed = input.trim()
  if (!trimmed) return undefined
  const slash = trimmed.indexOf("/")
  if (slash <= 0 || slash === trimmed.length - 1) return undefined
  return {
    providerID: trimmed.slice(0, slash),
    modelID: trimmed.slice(slash + 1),
  }
}

export default DelegatedAccess
