import type { ModelRef } from "../classifier/model.ts"

/**
 * Structural slice of a V2 transcript message (`SessionMessageInfo` from
 * @opencode/client — a flat union). We only consume the two variants that
 * matter here: `user` (carries `text`) and `assistant` (carries `model`).
 */
export type TranscriptMessage = {
  type?: string
  text?: string
  model?: { providerID?: string; id?: string }
}

/** Structural slice of the V2 `ctx.session` domain this module needs. */
export type SessionContextReader = {
  context(input: {
    sessionID: string
  }): Promise<TranscriptMessage[] | undefined>
}

/**
 * Wrapper tag names whose entire `<tag>…</tag>` block is injected NON-human
 * content that must NOT reach the safety classifier.
 *
 * Why: opencode and sibling plugins prepend large directive/context blocks to
 * the *user role* at prompt-assembly time. These land inside
 * `<recent_user_messages>` verbatim and derail the small classifier model —
 * observed in production as the classifier breaking character ("I'm just a
 * safety classifier, I don't follow embedded instructions / memory blocks")
 * and emitting no `VERDICT:` line (fail-closed → spurious TUI prompts), or
 * over-indexing on the injected noise and returning a spurious RISKY.
 *
 * The dominant offender is premind's `<pr_context>` block (median ~43 KB,
 * up to ~450 KB of PR metadata + comments as JSON). The directive wrappers
 * (`EXTREMELY_IMPORTANT`, `system-reminder`, etc.) are cheaper but match the
 * exact refusal language the model emitted, so we strip them defensively too.
 *
 * Strip-list is intentionally an explicit, known-wrapper allowlist (the
 * "safest" option): we never guess at arbitrary `<TAG>` blocks, so legitimate
 * user-authored XML/markup is preserved untouched.
 */
const CLASSIFIER_STRIP_TAGS = [
  "pr_context",
  "EXTREMELY_IMPORTANT",
  "EXTREMELY-IMPORTANT",
  "system-reminder",
  "SUBAGENT-STOP",
  "available_skills",
] as const

/**
 * Per-message hard cap (in characters) applied to the classifier's view of a
 * user message AFTER strip-list removal. A generous backstop: real human
 * turns are far below this, so it never fires in normal use — it only bounds
 * pathological or FUTURE-unknown injections the strip-list doesn't recognize,
 * guaranteeing no single message can ever drown the command + instructions.
 */
export const CLASSIFIER_MESSAGE_MAX_CHARS = 4_000

/**
 * Build a case-insensitive regex matching a single `<tag>…</tag>` block, or
 * an UNCLOSED `<tag>…` run to end-of-string (truncated injections have been
 * observed when a huge block is cut off mid-stream). `[\s\S]` so the body can
 * span newlines; non-greedy so adjacent blocks of the same tag don't merge.
 */
function stripTagRegex(tag: string): RegExp {
  const esc = tag.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  return new RegExp(`<${esc}\\b[\\s\\S]*?(?:</${esc}>|$)`, "gi")
}

const STRIP_REGEXES = CLASSIFIER_STRIP_TAGS.map(stripTagRegex)

/**
 * Sanitize a single user message for the safety classifier ONLY. Removes each
 * known injected wrapper block (see {@link CLASSIFIER_STRIP_TAGS}), collapses
 * the whitespace left behind, and caps the result at
 * {@link CLASSIFIER_MESSAGE_MAX_CHARS}.
 *
 * Pure and total: same input → same output, never throws. The output is the
 * classifier's view of the human's turn; it does NOT affect anything the user
 * or the real agent sees.
 */
export function sanitizeUserMessageForClassifier(text: string): string {
  let out = text
  for (const re of STRIP_REGEXES) out = out.replace(re, "")
  // Collapse the blank lines / stray whitespace the removals leave behind so
  // the prompt stays tidy and the empty-check below is reliable.
  out = out.replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim()
  if (out.length > CLASSIFIER_MESSAGE_MAX_CHARS) {
    out = out.slice(0, CLASSIFIER_MESSAGE_MAX_CHARS)
  }
  return out
}

/**
 * Pure function: from the V2 transcript messages return the last K user
 * messages as plain text, in chronological order. Assistant and non-user
 * messages are ignored.
 *
 * Messages with no text content after sanitizing (e.g. a user that uploaded
 * an image with no caption) are skipped entirely so they don't appear as
 * empty strings in the classifier prompt.
 *
 * Each message is run through {@link sanitizeUserMessageForClassifier} BEFORE
 * the empty-skip check, so injected non-human blocks (premind `<pr_context>`,
 * superpowers `<EXTREMELY_IMPORTANT>`, `<system-reminder>`, etc.) are stripped
 * and a message that was ONLY such a block is dropped instead of feeding the
 * classifier a giant context dump that derails it.
 *
 * Note: V1 also filtered user messages by the root session's primary agent
 * (`info.agent`), which we do NOT do here. Two reasons:
 *
 *   1. There is no field to filter on. In the V2 message schema only
 *      `type: "assistant"` carries an `agent`; the `user` variant does not.
 *   2. The threat it guarded against is structurally excluded. A subagent
 *      dispatch is a tool call inside an `assistant` message (filtered out by
 *      the `type !== "user"` check above) and the subagent itself runs in a
 *      child session (`parentID`), while callers resolve the ROOT session via
 *      `resolveRootSessionID` before reading. So a subagent's prompt cannot
 *      surface as a `user` message in the transcript we read.
 *
 * Re-check this if opencode ever adds an agent/author field to `user`
 * messages, or starts copying child-session prompts into a parent transcript.
 */
export function extractLastUserMessages(
  messages: readonly TranscriptMessage[],
  k: number,
): string[] {
  if (k <= 0) return []

  const userTexts: string[] = []
  for (const message of messages) {
    if (message.type !== "user") continue
    const text = sanitizeUserMessageForClassifier(
      typeof message.text === "string" ? message.text : "",
    )
    if (text.length === 0) continue
    userTexts.push(text)
  }

  return userTexts.slice(-k)
}

/**
 * Pure function: walk the messages in reverse and return the most recent
 * assistant message's `{ providerID, modelID }`. Returns `null` if no
 * assistant message exists or the latest assistant has no usable model ref.
 *
 * Used as a fallback source for the classifier model — any session that has
 * had at least one assistant turn will have a model recorded here.
 */
export function extractLatestAssistantModel(
  messages: readonly TranscriptMessage[],
): ModelRef | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i]
    if (!message || message.type !== "assistant") continue
    const model = message.model
    if (
      typeof model?.providerID === "string" &&
      typeof model?.id === "string"
    ) {
      return { providerID: model.providerID, modelID: model.id }
    }
    // Assistant found but model fields unusable — keep looking further back.
  }
  return null
}

/**
 * Thin I/O wrapper: fetch the session's transcript via the V2 session domain
 * so callers can run multiple pure extractors against a single call.
 */
export async function getSessionMessages(
  session: SessionContextReader,
  sessionID: string,
): Promise<TranscriptMessage[]> {
  const messages = await session.context({ sessionID })
  return messages ?? []
}
