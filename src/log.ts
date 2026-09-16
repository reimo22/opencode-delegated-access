import { appendFile } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"

/**
 * File- and console-backed logger.
 *
 * V2 exposes no `client.app.log` domain, and the daemonized V2 service runs
 * with stdout on `/dev/null`, so a console-only logger is invisible in
 * production. Every entry is therefore ALSO appended to
 * `<state>/opencode/delegated-access.log`, which is greppable:
 *
 *     tail -f ~/.local/state/opencode/delegated-access.log
 *
 * The console sink is kept because it is visible when the plugin is loaded by
 * a foreground client (`--standalone`, tests).
 *
 * Semantics: fire-and-forget, never throws. The logger's whole job is
 * diagnostics, so a failed write must not take down a permission decision —
 * that contract is why the catch below is deliberately broad.
 */
export type Logger = {
  debug: (message: string, extra?: Record<string, unknown>) => void
  info: (message: string, extra?: Record<string, unknown>) => void
  warn: (message: string, extra?: Record<string, unknown>) => void
  error: (message: string, extra?: Record<string, unknown>) => void
}

/** Service name embedded in every log entry from this plugin. */
export const LOG_SERVICE = "delegated-access"

const CONSOLE_PREFIX = "[delegated-access]"

/** Default log file, following XDG when the environment sets it. */
export function defaultLogPath(): string {
  const stateHome =
    process.env.XDG_STATE_HOME ?? join(homedir(), ".local", "state")
  return join(stateHome, "opencode", `${LOG_SERVICE}.log`)
}

function format(
  level: string,
  message: string,
  extra: Record<string, unknown> | undefined,
): string {
  const extraPart = extra ? ` ${JSON.stringify(extra)}` : ""
  return `${CONSOLE_PREFIX} ${level} ${message}${extraPart}`
}

function writeLine(path: string, line: string): void {
  // ponytail: unbounded append, one file per location. Rotate (size or mtime
  // cap) when this file actually gets large enough to matter.
  try {
    appendFile(path, `${new Date().toISOString()} ${line}\n`, () => {})
  } catch {
    // Never throw from the logger.
  }
}

/** Create a logger writing to `path` (and console). */
export function createLogger(path: string = defaultLogPath()): Logger {
  const emit = (
    level: "debug" | "info" | "warn" | "error",
    consoleFn: (...args: unknown[]) => void,
    message: string,
    extra?: Record<string, unknown>,
  ): void => {
    const line = format(level, message, extra)
    consoleFn(line)
    writeLine(path, line)
  }

  return {
    debug: (message, extra) =>
      emit("debug", console.log, message, extra),
    info: (message, extra) => emit("info", console.log, message, extra),
    warn: (message, extra) => emit("warn", console.warn, message, extra),
    error: (message, extra) => emit("error", console.error, message, extra),
  }
}
