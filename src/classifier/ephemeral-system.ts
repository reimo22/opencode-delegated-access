/**
 * Registry + transform for isolating the ephemeral classifier session's
 * system prompt from opencode's global instruction context.
 *
 * ## Why this exists
 *
 * `session.prompt`'s `system` field is ADDED to opencode's assembled system
 * context, not a replacement (confirmed against opencode 1.15.x). That means
 * the ephemeral classifier session receives, in addition to our classifier
 * system prompt:
 *
 *   - the default agent's prompt,
 *   - the model-specific provider prompt, and
 *   - the user's GLOBAL INSTRUCTIONS (AGENTS.md, instruction files, and any
 *     skill/preamble injected there — e.g. a "you MUST invoke the
 *     using-superpowers skill before ANY response" directive).
 *
 * Those global instructions out-muscle our classifier prompt: the small
 * classifier model starts behaving like a full agent ("Let me invoke the
 * using-superpowers skill…") and never emits a `VERDICT:` line, so
 * `parseVerdict` fails and the classification falls closed. Observed in
 * production as repeated `classifier: response did not parse` warnings.
 *
 * V2 assembles a separate request per request kind, so isolation must be
 * registered per kind via {@link registerEphemeralIsolationHooks}. For our
 * ephemeral classifier sessions ONLY, we replace the assembled system with
 * just the classifier system prompt — stripping the polluting global context
 * so the classifier sees nothing but its own instructions.
 */

/**
 * Maps an ephemeral classifier session ID to the exact system prompt that
 * session's classification should use. Populated by the classify path right
 * after it creates the ephemeral session, read by the system-transform hook,
 * and cleared when the session is deleted.
 */
export class EphemeralSystemRegistry {
  private readonly _map = new Map<string, string>()

  set(sessionID: string, systemPrompt: string): void {
    this._map.set(sessionID, systemPrompt)
  }

  get(sessionID: string): string | undefined {
    return this._map.get(sessionID)
  }

  has(sessionID: string): boolean {
    return this._map.has(sessionID)
  }

  delete(sessionID: string): void {
    this._map.delete(sessionID)
  }
}

type V2SessionRequest = {
  sessionID: string
  system: Array<{ type: string; text: string }>
  tools: Record<string, unknown>
}

type V2SessionHooks = {
  hook(
    name: "context" | "generate",
    callback: (event: V2SessionRequest) => void,
  ): Promise<unknown>
}

/** Isolate both request kinds the classifier path can produce: the agent-loop
 * `context` request and the transient `generate` request issued by
 * `session.generate`. Registering only `context` leaves the classifier prompt
 * unapplied on the generate path (the 2026-09-16 fail-closed bug). */
export async function registerEphemeralIsolationHooks(
  session: V2SessionHooks,
  sessionIDs: ReadonlySet<string>,
  registry: EphemeralSystemRegistry,
  onIsolated?: (sessionID: string) => void,
): Promise<void> {
  const isolate = (event: V2SessionRequest): void => {
    if (!sessionIDs.has(event.sessionID)) return
    const systemPrompt = registry.get(event.sessionID)
    if (systemPrompt === undefined) return
    event.system = [{ type: "text", text: systemPrompt }]
    event.tools = {}
    onIsolated?.(event.sessionID)
  }

  await session.hook("context", isolate)
  await session.hook("generate", isolate)
}
