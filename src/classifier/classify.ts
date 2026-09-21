import {
  CLASSIFIER_SYSTEM_PROMPT,
  buildClassifierUserPrompt,
} from "./prompt.ts"
import {
  DIRECTORY_CLASSIFIER_SYSTEM_PROMPT,
  buildDirectoryClassifierUserPrompt,
} from "./prompt.ts"
import { parseVerdict, type Verdict } from "./parse.ts"
import type { ModelRef } from "./model.ts"
import type { RepoContext, DualRepoContext } from "../repo-context.ts"
import type { ApprovalEntry } from "../permission/approval-history.ts"
import type { Logger } from "../log.ts"

/**
 * Structural slice of the V2 `ctx.generate` domain the classifier needs.
 * (Structural typing keeps us resilient to minor shape drift — the V2
 * package root doesn't export domain types directly.)
 *
 * `generate.text` is a one-shot model call: it creates no session, invokes no
 * tools, and adds nothing to session history. That is why the classifier uses
 * it — a session-per-classification used to flood the session list with
 * `[delegated-access classifier]` entries.
 */
export type ClassifierGenerator = {
  text(
    input: { prompt: string; model?: { providerID: string; id: string } },
    requestOptions?: { signal?: AbortSignal },
  ): Promise<{ text?: string } | undefined>
}

/**
 * Run the safety classifier for a permission subject (a bash command, a
 * directory path, or any future permission type) and return a verdict.
 *
 * Callers supply the LLM system prompt and a user-prompt builder so this
 * function remains agnostic about what is being classified.
 *
 * Flow:
 *   1. Build the classifier prompt: the caller-supplied system prompt is
 *      prepended to the user prompt built from subject + recent user messages.
 *      There is no session and no system-prompt channel, so the system
 *      instructions travel inline.
 *   2. Call `generate.text` with the classifier model attached. Because
 *      `generate.text` never dispatches tools and never runs the agent loop,
 *      the classifier cannot inherit opencode's global agent
 *      preamble/instructions, and no session is created.
 *   3. Parse the response text with {@link parseVerdict}.
 *   4. On timeout, abort the in-flight request and treat the attempt as failed
 *      (fail-closed — a partial response is NEVER trusted).
 *
 * Fail-closed behaviour: returns `null` for any error, malformed response,
 * or timeout exceeding `timeoutMs`. Callers should treat `null` as "classifier
 * failure → fall back to the normal opencode approval prompt".
 */
export async function classifySubject(args: {
  generate: ClassifierGenerator
  /** The string being classified (command, path pattern, etc.). */
  subject: string
  /** Recent human-authored messages to give the classifier context. */
  userMessages: string[]
  /** Session ID the permission originated from (for logging/context only). */
  parentSessionID: string
  model: ModelRef
  timeoutMs: number
  /** LLM system prompt for this permission type. */
  systemPrompt: string
  /**
   * Builds the user-turn prompt from `subject` + `userMessages`.
   * Called exactly once per invocation with the same `subject`/`userMessages`
   * passed to this function.
   */
  buildUserPrompt: (args: {
    subject: string
    userMessages: string[]
    repoContext?: DualRepoContext | RepoContext | null
    priorApprovals?: ApprovalEntry[]
  }) => string
  /**
   * Optional repo context handed to the classifier as additional
   * decision-shaping signal. Accepts either the legacy single-snapshot
   * `RepoContext` or the newer `DualRepoContext` (session-pinned + live)
   * for PR-scoped elevated trust. Rendered into the prompt by the
   * caller-supplied builder. `null` means "unavailable" (not a git repo,
   * gh missing, etc.) and is rendered as no <repo_context> block.
   */
  repoContext?: DualRepoContext | RepoContext | null
  /**
   * Pre-sorted (newest first) list of recent human approval/rejection
   * decisions to surface to the classifier as prior-decision evidence.
   * Forwarded verbatim to `buildUserPrompt`. Empty / undefined → no
   * `<prior_human_approvals>` block in the prompt.
   */
  priorApprovals?: ApprovalEntry[]
  /**
   * Optional diagnostic logger. Every fail-closed branch (prompt error,
   * timeout, empty response, unparseable verdict) emits an actionable log
   * line so an upstream API break isn't silently swallowed by the fail-closed
   * `catch`. When omitted, failures are silent (preserves the historical
   * behaviour for callers that don't pass a logger).
   */
  log?: Logger
  /**
   * Number of extra attempts to make if the classifier prompt TIMES OUT.
   * `0` (default) = single attempt, no retry. Only timeouts retry — other
   * failures (unparseable verdict, thrown prompt) are returned immediately,
   * since retrying them just wastes time. Each retry uses the FULL `timeoutMs`.
   */
  retries?: number
  /**
   * Called exactly once with the FINAL failure class when classification
   * ultimately fails (after any retries). Not called on success. Lets the
   * caller surface a notification distinguishing a transient timeout from a
   * harder error. `"timeout"` = the prompt(s) timed out; `"error"` =
   * anything else (thrown prompt, empty/unparseable response).
   */
  onFailure?: (failureClass: ClassifyFailureClass) => void
}): Promise<Verdict | null> {
  const { retries = 0, onFailure } = args

  // Retry loop: a `timeout` OR a `malformed` (unparseable) outcome is retried
  // (up to `retries` times). Hard errors (thrown prompt, empty response) are
  // final immediately — retrying them just wastes time. The full `timeoutMs`
  // is used on every attempt; the success case returns fast regardless of the
  // timeout ceiling.
  //
  // A retry that FOLLOWS a malformed response asks the model again with an
  // explicit format-correction instruction appended — small models that
  // narrate their role instead of emitting a VERDICT line usually comply on
  // the second, blunter ask.
  const maxAttempts = Math.max(0, retries) + 1
  let lastFailure: ClassifyFailureClass = "error"
  let priorWasMalformed = false
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const outcome = await classifyOnce(args, attempt, maxAttempts, priorWasMalformed)
    if (outcome.kind === "verdict") return outcome.verdict
    // `malformed` is reported to callers as the "error" failure class — the
    // public ClassifyFailureClass surface stays timeout|error.
    lastFailure = outcome.kind === "timeout" ? "timeout" : "error"
    priorWasMalformed = outcome.kind === "malformed"
    if (outcome.kind !== "timeout" && outcome.kind !== "malformed") break
  }
  onFailure?.(lastFailure)
  return null
}

/** Final failure category reported to {@link classifySubject}'s `onFailure`. */
export type ClassifyFailureClass = "timeout" | "error"

/** Internal per-attempt outcome of the classifier. */
type ClassifyOutcome =
  | { kind: "verdict"; verdict: Verdict }
  | { kind: "timeout" }
  /** Prompt returned text, but it had no parseable VERDICT line. Retryable. */
  | { kind: "malformed" }
  /** Hard failure (create error, thrown prompt, empty response). Not retried. */
  | { kind: "error" }

/**
 * Instruction appended to the user prompt on a retry that follows a malformed
 * (unparseable) response. Kept blunt and format-only so a small model that
 * narrated its role on the first attempt answers correctly the second time.
 * Wrapped in a delimiter so it reads as a distinct correction, not part of the
 * original subject.
 */
const FORMAT_CORRECTION_INSTRUCTION = `

<format_correction>
Your previous response did not match the required format.
Please answer ONLY in this exact format, with no other text:
VERDICT: <SAFE|RISKY>
REASON: <one short sentence>
Classify the original subject again now.
</format_correction>`

/**
 * A single classifier attempt: build the prompt, call `generate.text` with a
 * timeout, parse the verdict. Returns a discriminated outcome so the caller's
 * retry loop can distinguish a retryable timeout from a final error. Never
 * throws.
 */
async function classifyOnce(
  args: Parameters<typeof classifySubject>[0],
  attempt: number,
  maxAttempts: number,
  /**
   * When true, this attempt follows a malformed response: append
   * {@link FORMAT_CORRECTION_INSTRUCTION} to the user prompt so the model is
   * explicitly told to fix its output format.
   */
  correctFormat = false,
): Promise<ClassifyOutcome> {
  const {
    generate,
    subject,
    userMessages,
    parentSessionID,
    model,
    timeoutMs,
    systemPrompt,
    buildUserPrompt,
    repoContext,
    priorApprovals,
    log,
  } = args
  void parentSessionID

  let timedOut = false
  try {
    // Step 1: assemble the prompt. `generate.text` has no system-prompt
    // channel, so the classifier system instructions are prepended inline.
    const baseUserPrompt = buildUserPrompt({
      subject,
      userMessages,
      repoContext: repoContext ?? null,
      priorApprovals: priorApprovals ?? [],
    })
    const userPrompt = correctFormat
      ? baseUserPrompt + FORMAT_CORRECTION_INSTRUCTION
      : baseUserPrompt
    const prompt = `${systemPrompt}\n\n${userPrompt}`

    // Step 2: one-shot model call with a timeout. The signal aborts the
    // in-flight HTTP request when the timeout fires.
    const controller = new AbortController()
    const generateCall = generate.text(
      { prompt, model: { providerID: model.providerID, id: model.modelID } },
      { signal: controller.signal },
    )

    const response = await withTimeout(generateCall, timeoutMs, () => {
      timedOut = true
      controller.abort()
    })

    // Fail-closed gate: if the timeout fired at ANY point during the race,
    // discard whatever the promise returned. Partial pre-abort responses have
    // been observed to contain well-formed "VERDICT: SAFE" text that would
    // otherwise auto-approve a command whose classification never actually
    // completed — violating the plugin's fail-closed contract (see README
    // "How it's safe").
    if (timedOut) {
      log?.warn("classifier: timeout — no verdict (fail-closed)", {
        timeoutMs,
        attempt,
        maxAttempts,
        willRetry: attempt < maxAttempts,
      })
      return { kind: "timeout" }
    }

    if (!response || typeof response.text !== "string" || response.text.length === 0) {
      log?.warn("classifier: prompt returned no response (fail-closed)", {})
      return { kind: "error" }
    }

    // Step 3: parse.
    const text = response.text
    const verdict = parseVerdict(text)
    if (!verdict) {
      // Surface the raw model text (truncated) so an output-format break —
      // e.g. the model narrating its role and never emitting a VERDICT line —
      // is debuggable instead of a silent fail-closed. Returned as
      // `malformed` (not `error`) so the retry loop gives it a second,
      // format-corrected attempt before failing closed.
      log?.warn("classifier: response did not parse to a verdict (malformed)", {
        rawTextPreview: text.slice(0, 500),
        rawTextLength: text.length,
        attempt,
        maxAttempts,
        willRetry: attempt < maxAttempts,
        wasFormatCorrected: correctFormat,
      })
      return { kind: "malformed" }
    }
    return { kind: "verdict", verdict }
  } catch (e) {
    log?.error("classifier: prompt threw (fail-closed)", {
      error: e instanceof Error ? e.message : String(e),
    })
    return { kind: "error" }
  }
}

/**
 * Convenience wrapper around {@link classifySubject} that supplies the
 * bash-specific system prompt and user-prompt builder. Preserved so
 * existing call-sites in handler.ts need no changes.
 */
export function classifyCommand(
  args: Omit<
    Parameters<typeof classifySubject>[0],
    "subject" | "systemPrompt" | "buildUserPrompt"
  > & { command: string },
): ReturnType<typeof classifySubject> {
  const { command, ...rest } = args
  return classifySubject({
    ...rest,
    subject: command,
    systemPrompt: CLASSIFIER_SYSTEM_PROMPT,
    buildUserPrompt: ({ subject, userMessages, repoContext, priorApprovals }) =>
      buildClassifierUserPrompt({
        command: subject,
        userMessages,
        repoContext: repoContext ?? null,
        priorApprovals: priorApprovals ?? [],
      }),
  })
}

/**
 * Convenience wrapper around {@link classifySubject} for external-directory
 * permissions: supplies the directory-specific system prompt and user-prompt
 * builder.
 */
export function classifyDirectory(
  args: Omit<
    Parameters<typeof classifySubject>[0],
    "subject" | "systemPrompt" | "buildUserPrompt"
  > & { path: string },
): ReturnType<typeof classifySubject> {
  const { path, ...rest } = args
  return classifySubject({
    ...rest,
    subject: path,
    systemPrompt: DIRECTORY_CLASSIFIER_SYSTEM_PROMPT,
    buildUserPrompt: ({ subject, userMessages, repoContext, priorApprovals }) =>
      buildDirectoryClassifierUserPrompt({
        subject,
        userMessages,
        repoContext: repoContext ?? null,
        priorApprovals: priorApprovals ?? [],
      }),
  })
}

/**
 * Race a promise against a timeout. If the timeout fires first, runs
 * `onTimeout` (so callers can abort in-flight work) and then resolves to
 * `null`. Otherwise passes through the promise's result.
 */
async function withTimeout<T>(
  p: Promise<T>,
  timeoutMs: number,
  onTimeout: () => Promise<void> | void,
): Promise<T | null> {
  let timer: NodeJS.Timeout | undefined
  const timeout = new Promise<null>((resolve) => {
    timer = setTimeout(async () => {
      try {
        await onTimeout()
      } catch {
        // Timeout handler errors are swallowed — we're on the failure path.
      }
      resolve(null)
    }, timeoutMs)
  })
  try {
    const result = await Promise.race([p, timeout])
    return result as T | null
  } finally {
    if (timer) clearTimeout(timer)
  }
}
