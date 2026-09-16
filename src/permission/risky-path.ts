import { sendNotification } from "../notify/notify.ts"
import type { Logger } from "../log.ts"

/**
 * Programmatic permission resolver injected by the caller. Resolves the
 * pending permission request (finding its requestID via the permission
 * domain, matching session/action/resources) and replies to it. Swallows
 * its own errors — the TUI prompt remains as a fallback.
 */
export type PermissionReplier = (
  response: "once" | "always" | "reject",
) => Promise<void>

/** Upper bound on the command string we embed in the notification body. */
const COMMAND_DISPLAY_MAX = 180

/** Button labels used for the RISKY notification. */
const APPROVE_LABEL = "Approve"
const REJECT_LABEL = "Reject"

/**
 * Drive the RISKY-path notification _in the background_, alongside opencode's
 * normal TUI permission prompt.
 *
 * Called AFTER the plugin's permission evaluate hook has resolved with
 * "ask" (no override), so opencode is already showing its in-TUI prompt.
 * This function fires and awaits the notification independently:
 *
 *   - If the user clicks **Approve** in the notification, we resolve the
 *     permission with `response: "once"` via the injected replier — this
 *     closes the TUI prompt programmatically and opencode proceeds.
 *   - If they click **Reject**, we resolve with `response: "reject"` —
 *     same deal, but opencode blocks the command.
 *   - Any other outcome (timeout, cancel, body click, notifier error, unknown
 *     action label) is a no-op: the TUI prompt is still live and the user
 *     can respond there as normal.
 *
 * Replier errors are swallowed — the TUI prompt remains as a fallback, so a
 * transient failure doesn't leave the user stranded.
 *
 * This function is expected to be called with fire-and-forget semantics; it
 * never returns anything useful and never throws.
 */
export async function runRiskyPathInBackground(args: {
  reply: PermissionReplier
  command: string
  reason: string
  sound: boolean
  timeoutSec: number
  /**
   * Records the notification's outcome. Without this the notifier is a black
   * box: it is fire-and-forget, so "was the user ever actually notified?" is
   * unanswerable from the outside.
   */
  log?: Logger
}): Promise<void> {
  const { reply, command, reason, sound, timeoutSec } = args

  const displayCmd =
    command.length > COMMAND_DISPLAY_MAX
      ? command.slice(0, COMMAND_DISPLAY_MAX) + "…"
      : command

  const displayReason = reason.length > 0 ? ` (${reason})` : ""

  const result = await sendNotification({
    title: "delegated-access: review risky command",
    message: `${displayCmd}${displayReason}`,
    actions: [APPROVE_LABEL, REJECT_LABEL],
    sound,
    timeoutSec,
  })

  if (result.type === "error") {
    // Degraded, not broken: the TUI prompt is still live. Worth a warning
    // because it silently removes the notification affordance.
    args.log?.warn("risky notification failed; TUI prompt remains", {
      error: result.error.message,
    })
  } else {
    args.log?.debug("risky notification outcome", {
      outcome: result.type,
      ...(result.type === "action" ? { label: result.label } : {}),
    })
  }

  if (result.type !== "action") return

  let response: "once" | "reject" | undefined
  if (result.label === APPROVE_LABEL) response = "once"
  else if (result.label === REJECT_LABEL) response = "reject"
  if (!response) return

  try {
    // Resolve the permission programmatically; this closes the TUI prompt.
    await reply(response)
  } catch {
    // Swallow — TUI prompt is still live as a fallback.
  }
}
