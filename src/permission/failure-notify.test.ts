import { describe, it, expect, vi } from "vitest"
import {
  FailureNotifyRateLimiter,
  runFailureNotificationInBackground,
} from "./failure-notify.ts"
import type { NotifyActionResult } from "../notify/notify.ts"
import { makeLogger } from "../testing/v2-fixtures.ts"

describe("FailureNotifyRateLimiter", () => {
  it("allows the first failure", () => {
    const rl = new FailureNotifyRateLimiter({ cooldownMs: 1000 })
    const d = rl.register("timeout", 0)
    expect(d.notify).toBe(true)
    expect(d.suppressedCount).toBe(0)
  })

  it("suppresses a second failure within the cooldown window", () => {
    const rl = new FailureNotifyRateLimiter({ cooldownMs: 1000 })
    rl.register("timeout", 0)
    const d = rl.register("timeout", 500)
    expect(d.notify).toBe(false)
  })

  it("allows again once the cooldown has elapsed", () => {
    const rl = new FailureNotifyRateLimiter({ cooldownMs: 1000 })
    rl.register("timeout", 0)
    rl.register("timeout", 500) // suppressed
    const d = rl.register("timeout", 1001)
    expect(d.notify).toBe(true)
  })

  it("reports how many failures were suppressed since the last allowed notify (burst collapse)", () => {
    const rl = new FailureNotifyRateLimiter({ cooldownMs: 1000 })
    rl.register("timeout", 0) // allowed
    rl.register("timeout", 100) // suppressed (1)
    rl.register("timeout", 200) // suppressed (2)
    rl.register("error", 300) // suppressed (3)
    const d = rl.register("timeout", 1001) // allowed again
    expect(d.notify).toBe(true)
    expect(d.suppressedCount).toBe(3)
  })

  it("disables the rate limit entirely when cooldownMs is 0", () => {
    const rl = new FailureNotifyRateLimiter({ cooldownMs: 0 })
    expect(rl.register("timeout", 0).notify).toBe(true)
    expect(rl.register("timeout", 0).notify).toBe(true)
    expect(rl.register("timeout", 1).notify).toBe(true)
  })
})

/** V2 injects a replier instead of exposing the SDK client. */
function makeReply(impl?: () => Promise<void>) {
  return vi.fn(impl ?? (async () => {}))
}

const baseArgs = {
  command: "whoami && date",
  failureClass: "timeout" as const,
  suppressedCount: 0,
  sound: false,
  timeoutSec: 30,
}

describe("runFailureNotificationInBackground", () => {
  it("sends a notification with a Reject action but NO Approve action", async () => {
    const sent: Parameters<typeof import("../notify/notify.ts").sendNotification>[0][] =
      []
    const sendNotification = vi.fn(async (a: { actions?: string[] }) => {
      sent.push(a as never)
      return { type: "timeout" } as NotifyActionResult
    })
    const reply = makeReply()

    await runFailureNotificationInBackground({
      ...baseArgs,
      reply,
      sendNotification: sendNotification as never,
    })

    expect(sendNotification).toHaveBeenCalledTimes(1)
    const actions = sent[0]?.actions ?? []
    expect(actions).toContain("Reject")
    expect(actions).not.toContain("Approve")
  })

  it("resolves the permission as 'reject' when the user clicks Reject", async () => {
    const sendNotification = vi.fn(
      async () => ({ type: "action", label: "Reject" }) as NotifyActionResult,
    )
    const reply = makeReply()

    await runFailureNotificationInBackground({
      ...baseArgs,
      reply,
      sendNotification: sendNotification as never,
    })

    expect(reply).toHaveBeenCalledTimes(1)
    expect(reply).toHaveBeenCalledWith("reject")
  })

  it("does NOT resolve the permission on timeout/cancel/click (TUI prompt stays)", async () => {
    const reply = makeReply()
    for (const result of [
      { type: "timeout" },
      { type: "cancel" },
      { type: "click" },
    ] as NotifyActionResult[]) {
      const sendNotification = vi.fn(async () => result)
      await runFailureNotificationInBackground({
        ...baseArgs,
        reply,
        sendNotification: sendNotification as never,
      })
    }
    expect(reply).not.toHaveBeenCalled()
  })

  it("mentions the failure class in the notification title", async () => {
    const titles: string[] = []
    const sendNotification = vi.fn(async (a: { title: string }) => {
      titles.push(a.title)
      return { type: "timeout" } as NotifyActionResult
    })
    const reply = makeReply()

    await runFailureNotificationInBackground({
      ...baseArgs,
      failureClass: "timeout",
      reply,
      sendNotification: sendNotification as never,
    })

    expect(titles[0]?.toLowerCase()).toMatch(/timed out|timeout/)
  })

  it("mentions a collapsed burst count when suppressedCount > 0", async () => {
    const messages: string[] = []
    const sendNotification = vi.fn(async (a: { message: string }) => {
      messages.push(a.message)
      return { type: "timeout" } as NotifyActionResult
    })
    const reply = makeReply()

    await runFailureNotificationInBackground({
      ...baseArgs,
      suppressedCount: 3,
      reply,
      sendNotification: sendNotification as never,
    })

    // 3 suppressed + this one = 4 total failures referenced.
    expect(messages[0]).toMatch(/4|3 more|\+3/)
  })

  it("swallows reply errors (TUI prompt remains as fallback)", async () => {
    const reply = makeReply(async () => {
      throw new Error("reply boom")
    })
    const sendNotification = vi.fn(
      async () => ({ type: "action", label: "Reject" }) as NotifyActionResult,
    )

    await expect(
      runFailureNotificationInBackground({
        ...baseArgs,
        reply,
        sendNotification: sendNotification as never,
      }),
    ).resolves.toBeUndefined()
  })

  it("records a notifier error as a warning, since that silently drops the notification affordance", async () => {
    const sendNotification = vi.fn(
      async () =>
        ({ type: "error", error: new Error("no notification daemon") }) as NotifyActionResult,
    )
    const reply = makeReply()
    const log = makeLogger()

    await runFailureNotificationInBackground({
      ...baseArgs,
      reply,
      sendNotification: sendNotification as never,
      log,
    })

    expect(reply).not.toHaveBeenCalled()
    expect(log.warn).toHaveBeenCalledWith(
      "classifier-failure notification failed; TUI prompt remains",
      { error: "no notification daemon" },
    )
  })
})
