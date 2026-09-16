/**
 * Console-backed logger. V2 of the plugin API exposes no `client.app.log`
 * domain, so diagnostics go through `console.*` — opencode captures plugin
 * stdout/stderr into its own log file, so entries are still greppable:
 *
 *     grep delegated-access ~/.local/share/opencode/log/*.log
 *
 * Semantics: fire-and-forget, never throws.
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

function format(
  level: string,
  message: string,
  extra: Record<string, unknown> | undefined,
): string {
  const extraPart = extra ? ` ${JSON.stringify(extra)}` : ""
  return `${CONSOLE_PREFIX} ${level} ${message}${extraPart}`
}

export function createLogger(): Logger {
  return {
    debug: (message, extra) => console.log(format("debug", message, extra)),
    info: (message, extra) => console.log(format("info", message, extra)),
    warn: (message, extra) => console.warn(format("warn", message, extra)),
    error: (message, extra) => console.error(format("error", message, extra)),
  }
}
