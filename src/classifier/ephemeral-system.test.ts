import { describe, it, expect } from "vitest"
import {
  EphemeralSystemRegistry,
  registerEphemeralIsolationHooks,
} from "./ephemeral-system.ts"

describe("EphemeralSystemRegistry", () => {
  it("returns undefined for an unregistered session", () => {
    const reg = new EphemeralSystemRegistry()
    expect(reg.get("sess_x")).toBeUndefined()
  })

  it("stores and retrieves a registered system prompt", () => {
    const reg = new EphemeralSystemRegistry()
    reg.set("sess_x", "CLASSIFIER PROMPT")
    expect(reg.get("sess_x")).toBe("CLASSIFIER PROMPT")
  })

  it("deletes a registered prompt", () => {
    const reg = new EphemeralSystemRegistry()
    reg.set("sess_x", "P")
    reg.delete("sess_x")
    expect(reg.get("sess_x")).toBeUndefined()
  })

  it("reports membership via has()", () => {
    const reg = new EphemeralSystemRegistry()
    expect(reg.has("sess_x")).toBe(false)
    reg.set("sess_x", "P")
    expect(reg.has("sess_x")).toBe(true)
  })
})

describe("registerEphemeralIsolationHooks", () => {
  it("isolates both agent-context and transient-generate requests", async () => {
    const registered = new Map<string, (event: any) => void>()
    const session = {
      hook: async (name: string, callback: (event: any) => void) => {
        registered.set(name, callback)
        return { dispose: async () => {} }
      },
    }
    const sessionIDs = new Set(["sess_eph"])
    const registry = new EphemeralSystemRegistry()
    registry.set("sess_eph", "classifier prompt")

    await registerEphemeralIsolationHooks(
      session,
      sessionIDs,
      registry,
    )

    for (const hookName of ["context", "generate"]) {
      const request = {
        sessionID: "sess_eph",
        system: [{ type: "text", text: "global instructions" }],
        tools: { shell: {} },
      }
      registered.get(hookName)?.(request)
      expect(request.system).toEqual([{ type: "text", text: "classifier prompt" }])
      expect(request.tools).toEqual({})
    }
  })

  it("leaves a non-classifier session's request untouched", async () => {
    const registered = new Map<string, (event: any) => void>()
    const session = {
      hook: async (name: string, callback: (event: any) => void) => {
        registered.set(name, callback)
        return { dispose: async () => {} }
      },
    }
    const registry = new EphemeralSystemRegistry()
    registry.set("sess_eph", "classifier prompt")

    await registerEphemeralIsolationHooks(session, new Set(["sess_eph"]), registry)

    for (const hookName of ["context", "generate"]) {
      const request = {
        sessionID: "sess_human",
        system: [{ type: "text", text: "global instructions" }],
        tools: { shell: {} },
      }
      registered.get(hookName)?.(request)
      expect(request.system).toEqual([{ type: "text", text: "global instructions" }])
      expect(request.tools).toEqual({ shell: {} })
    }
  })
})
