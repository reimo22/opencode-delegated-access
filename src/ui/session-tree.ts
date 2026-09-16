/**
 * Maximum number of `parentID` hops we'll follow before giving up. Real-world
 * subagent chains are typically 1-2 levels deep; this bound is generous
 * enough that we should never hit it in practice, and short enough that a
 * pathological mis-configuration can't chew through API calls.
 */
export const MAX_SESSION_PARENT_DEPTH = 10

/**
 * Structural slice of the V2 `ctx.session` domain this module needs.
 * (The V2 package root doesn't export domain types directly, and structural
 * typing keeps us resilient to minor shape drift.)
 */
export type SessionGetter = {
  get(input: { sessionID: string }): Promise<{ parentID?: string } | undefined>
}

/**
 * Walk a session's `parentID` chain up to the root session.
 *
 * Subagent dispatches in opencode create child sessions whose `parentID`
 * points at the dispatcher's session. To preserve the plugin's safety
 * property ("classifier only sees human messages"), bash permissions
 * originating inside a subagent must be classified against the ROOT
 * session's user messages — not the subagent's, whose "user" entries are
 * the dispatching agent's prompts.
 *
 * Fail-closed contract: returns `null` on ANY failure (session.get error,
 * missing payload, max depth exceeded, cycle detected). Callers MUST treat
 * `null` as "abort classification and leave the TUI prompt alone" so we
 * never auto-approve a command whose true chain-of-custody we couldn't
 * verify.
 */
export async function resolveRootSessionID(
  session: SessionGetter,
  sessionID: string,
): Promise<string | null> {
  const seen = new Set<string>()
  let current = sessionID

  for (let hops = 0; hops <= MAX_SESSION_PARENT_DEPTH; hops++) {
    // Cycle guard: if we've seen this ID before, the tree is malformed.
    if (seen.has(current)) return null
    seen.add(current)

    let info: { parentID?: string } | undefined
    try {
      info = await session.get({ sessionID: current })
    } catch {
      return null
    }
    if (!info) return null

    // No parent → we've reached the root.
    if (typeof info.parentID !== "string" || info.parentID.length === 0) {
      return current
    }

    current = info.parentID
  }

  // Exceeded MAX_SESSION_PARENT_DEPTH without finding a root.
  return null
}
