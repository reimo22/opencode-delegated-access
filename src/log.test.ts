import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { mkdtempSync, readFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { createLogger, defaultLogPath, LOG_SERVICE } from "./log.ts"

/** Wait until the fire-and-forget append has landed in the file. */
async function readLogLines(path: string): Promise<string[]> {
  const { vi } = await import("vitest")
  await vi.waitFor(() => {
    expect(readFileSync(path, "utf8").length).toBeGreaterThan(0)
  })
  return readFileSync(path, "utf8").trimEnd().split("\n")
}

describe("createLogger", () => {
  let dir: string
  let logPath: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "delegated-access-log-"))
    logPath = join(dir, "delegated-access.log")
  })

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true })
  })

  it("appends every level to the log file with the service prefix", async () => {
    const log = createLogger(logPath)

    log.debug("d-msg")
    log.info("i-msg")
    log.warn("w-msg")
    log.error("e-msg")

    const lines = await readLogLines(logPath)
    expect(lines).toHaveLength(4)
    expect(lines[0]).toContain(`[${LOG_SERVICE}] debug d-msg`)
    expect(lines[1]).toContain(`[${LOG_SERVICE}] info i-msg`)
    expect(lines[2]).toContain(`[${LOG_SERVICE}] warn w-msg`)
    expect(lines[3]).toContain(`[${LOG_SERVICE}] error e-msg`)
  })

  it("renders extra metadata as JSON on the same line", async () => {
    const log = createLogger(logPath)

    log.info("verdict parsed", { verdict: "SAFE", attempt: 1 })

    const lines = await readLogLines(logPath)
    expect(lines[0]).toContain('"verdict":"SAFE"')
    expect(lines[0]).toContain('"attempt":1')
  })

  it("keeps appending across calls rather than truncating", async () => {
    const log = createLogger(logPath)

    log.info("first")
    await readLogLines(logPath)
    log.info("second")

    const { vi } = await import("vitest")
    await vi.waitFor(async () => {
      const content = readFileSync(logPath, "utf8")
      expect(content).toContain("first")
      expect(content).toContain("second")
    })
  })

  it("never throws when the log path is unwritable", () => {
    const log = createLogger(join(dir, "missing-dir", "nested", "x.log"))

    expect(() => log.info("dropped")).not.toThrow()
  })
})

describe("defaultLogPath", () => {
  const original = process.env.XDG_STATE_HOME

  afterEach(() => {
    if (original === undefined) delete process.env.XDG_STATE_HOME
    else process.env.XDG_STATE_HOME = original
  })

  it("follows XDG_STATE_HOME when set", () => {
    process.env.XDG_STATE_HOME = "/tmp/xdg-state"

    expect(defaultLogPath()).toBe(
      join("/tmp/xdg-state", "opencode", `${LOG_SERVICE}.log`),
    )
  })

  it("falls back under the home directory when XDG_STATE_HOME is unset", () => {
    delete process.env.XDG_STATE_HOME

    expect(defaultLogPath()).toContain(
      join(".local", "state", "opencode", `${LOG_SERVICE}.log`),
    )
  })
})
