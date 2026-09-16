import { describe, it, expect, vi } from "vitest"
import { resolveRootSessionID } from "./session-tree.ts"

/**
 * Build a mock V2 session domain whose `get` resolves sessionIDs to their
 * `{ parentID }` shape from a provided map. IDs missing from the map throw a
 * "Not found" error so we can exercise the fail-closed path.
 *
 * Returns both the session domain and a spy on `get` so tests can assert on
 * the call sequence.
 */
function buildSession(parents: Record<string, string | undefined>) {
  const get = vi.fn(async ({ sessionID }: { sessionID: string }) => {
    if (!(sessionID in parents)) {
      throw new Error(`Session ${sessionID} not found`)
    }
    const parentID = parents[sessionID]
    return parentID !== undefined
      ? { id: sessionID, parentID }
      : { id: sessionID }
  })
  return {
    session: { get },
    get,
  }
}

describe("resolveRootSessionID", () => {
  it("returns the input sessionID when the session has no parent (root)", async () => {
    const { session, get } = buildSession({ root: undefined })
    const result = await resolveRootSessionID(session as never, "root")
    expect(result).toBe("root")
    expect(get).toHaveBeenCalledTimes(1)
  })

  it("walks up one level to find the root", async () => {
    const { session } = buildSession({
      child: "root",
      root: undefined,
    })
    const result = await resolveRootSessionID(session as never, "child")
    expect(result).toBe("root")
  })

  it("walks up multiple levels to find the root", async () => {
    const { session } = buildSession({
      leaf: "mid2",
      mid2: "mid1",
      mid1: "root",
      root: undefined,
    })
    const result = await resolveRootSessionID(session as never, "leaf")
    expect(result).toBe("root")
  })

  it("returns null when session.get throws at the starting session", async () => {
    const { session } = buildSession({})
    const result = await resolveRootSessionID(session as never, "missing")
    expect(result).toBeNull()
  })

  it("returns null when session.get throws mid-chain", async () => {
    const { session } = buildSession({
      leaf: "mid",
      // mid intentionally absent from the map → throws
    })
    const result = await resolveRootSessionID(session as never, "leaf")
    expect(result).toBeNull()
  })

  it("returns null when the chain exceeds the max depth of 10", async () => {
    // 12 levels of nesting: n11 → n10 → ... → n0 (root)
    const chain: Record<string, string | undefined> = {}
    for (let i = 11; i > 0; i--) chain[`n${i}`] = `n${i - 1}`
    chain.n0 = undefined
    const { session } = buildSession(chain)
    const result = await resolveRootSessionID(session as never, "n11")
    expect(result).toBeNull()
  })

  it("returns the root at exactly the max-depth boundary (10 hops)", async () => {
    // 10 hops: leaf → n9 → n8 → ... → n0 (root). 11 session.get calls total.
    const chain: Record<string, string | undefined> = {}
    for (let i = 9; i > 0; i--) chain[`n${i}`] = `n${i - 1}`
    chain.n0 = undefined
    chain.leaf = "n9"
    const { session, get } = buildSession(chain)
    const result = await resolveRootSessionID(session as never, "leaf")
    expect(result).toBe("n0")
    // 11 levels fetched: leaf, n9..n0.
    expect(get).toHaveBeenCalledTimes(11)
  })

  it("returns null on a cycle (session appears twice in the chain)", async () => {
    // A → B → A → ... (cycle)
    const { session } = buildSession({
      a: "b",
      b: "a",
    })
    const result = await resolveRootSessionID(session as never, "a")
    expect(result).toBeNull()
  })

  it("returns null on a self-cycle (parentID equals own ID)", async () => {
    const { session } = buildSession({ self: "self" })
    const result = await resolveRootSessionID(session as never, "self")
    expect(result).toBeNull()
  })

  it("treats a missing session payload as failure", async () => {
    const session = {
      get: vi.fn(async () => undefined),
    }
    const result = await resolveRootSessionID(session as never, "x")
    expect(result).toBeNull()
  })
})
