import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { mkdtempSync, readFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { createLogger, defaultLogPath, LOG_SERVICE } from "./log.ts"

/**
 * Wait until the log file contains every `expected` substring, then return its
 * contents. The logger appends fire-and-forget, so writes land asynchronously
 * and in unspecified order — tests must not assume a line index or count.
 */
async function waitForLog(path: string, expected: string[]): Promise<string> {
  const { vi } = await import("vitest")
  await vi.waitFor(() => {
    const content = readFileSync(path, "utf8")
    for (const needle of expected) {
      expect(content).toContain(needle)
    }
  })
  return readFileSync(path, "utf8")
}

/** Non-empty lines, so a trailing newline isn't counted as a line. */
function linesOf(content: string): string[] {
  return content.split("\n").filter((line) => line.length > 0)
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

    const content = await waitForLog(logPath, [
      `[${LOG_SERVICE}] debug d-msg`,
      `[${LOG_SERVICE}] info i-msg`,
      `[${LOG_SERVICE}] warn w-msg`,
      `[${LOG_SERVICE}] error e-msg`,
    ])

    // One line per call — no accidental extra writes or merging.
    expect(linesOf(content)).toHaveLength(4)
  })

  it("renders extra metadata as JSON on the same line", async () => {
    const log = createLogger(logPath)

    log.info("verdict parsed", { verdict: "SAFE", attempt: 1 })

    const content = await waitForLog(logPath, [
      '"verdict":"SAFE"',
      '"attempt":1',
    ])

    expect(linesOf(content)).toHaveLength(1)
    expect(linesOf(content)[0]).toContain(`[${LOG_SERVICE}] info verdict parsed`)
  })

  it("keeps appending across calls rather than truncating", async () => {
    const log = createLogger(logPath)

    log.info("first")
    await waitForLog(logPath, ["first"])
    log.info("second")

    const content = await waitForLog(logPath, ["first", "second"])
    expect(linesOf(content)).toHaveLength(2)
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
