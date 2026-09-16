import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { sendNotification } from "./notify.ts"
import type { NotifyActionResult } from "./notify.ts"

// Mock node-notifier: `default` is the cross-platform entry point (notify-send
// on Linux), `NotificationCenter` is the macOS-only constructor. Keeping both
// on the same module lets tests assert which backend a platform selects.
vi.mock("node-notifier", () => {
  const notify = vi.fn()
  const NotificationCenter = vi.fn()
  return {
    default: { notify, NotificationCenter },
    notify,
    NotificationCenter,
  }
})

import nn from "node-notifier"

type NotifyArgs = {
  title: string
  message: string
  sound?: boolean
  actions?: string | string[]
  closeLabel?: string
  timeout?: number | false
  wait?: boolean
}
type NotifyCallback = (
  err: Error | null,
  response: string,
  metadata?: { activationType?: string; activationValue?: string },
) => void

const crossPlatformEntry = nn as unknown as {
  notify: ReturnType<typeof vi.fn>
}
const NotificationCenter = (
  nn as unknown as { NotificationCenter: ReturnType<typeof vi.fn> }
).NotificationCenter

/** Install a scenario on the cross-platform backend and capture its options. */
function scenarioOnCrossPlatform(s: {
  callback?: (cb: NotifyCallback) => void
}) {
  const seen: NotifyArgs[] = []
  crossPlatformEntry.notify.mockImplementation(
    (options: NotifyArgs, cb: NotifyCallback) => {
      seen.push(options)
      s.callback?.(cb)
    },
  )
  return seen
}

/** Install a scenario on the macOS NotificationCenter backend. */
function scenarioOnNotificationCenter(s: {
  callback?: (cb: NotifyCallback) => void
}) {
  const seen: NotifyArgs[] = []
  NotificationCenter.mockImplementation(() => ({
    notify: (options: NotifyArgs, cb: NotifyCallback) => {
      seen.push(options)
      s.callback?.(cb)
    },
  }))
  return seen
}

const realPlatform = process.platform
function stubPlatform(value: string) {
  Object.defineProperty(process, "platform", { value, configurable: true })
}

beforeEach(() => {
  vi.clearAllMocks()
  stubPlatform("linux")
})

afterEach(() => {
  stubPlatform(realPlatform)
})

describe("sendNotification — macOS (NotificationCenter backend)", () => {
  beforeEach(() => {
    stubPlatform("darwin")
  })

  it("resolves with { type: 'action', label } when user clicks a button", async () => {
    scenarioOnNotificationCenter({
      callback: (cb) => {
        cb(null, "activate", {
          activationType: "actionClicked",
          activationValue: "Approve",
        })
      },
    })

    const result = await sendNotification({
      title: "Test",
      message: "hi",
      actions: ["Approve", "Reject"],
      timeoutSec: 30,
    })

    expect(result).toEqual<NotifyActionResult>({
      type: "action",
      label: "Approve",
    })
  })

  it("resolves with { type: 'cancel' } when user dismisses / closes", async () => {
    scenarioOnNotificationCenter({
      callback: (cb) => cb(null, "closed", { activationType: "closed" }),
    })

    const result = await sendNotification({
      title: "t",
      message: "m",
      actions: ["Approve", "Reject"],
    })

    expect(result.type).toBe("cancel")
  })

  it("resolves with { type: 'timeout' } when the notification times out", async () => {
    scenarioOnNotificationCenter({
      callback: (cb) => cb(null, "timeout", { activationType: "timeout" }),
    })

    const result = await sendNotification({ title: "t", message: "m" })
    expect(result.type).toBe("timeout")
  })

  it("resolves with { type: 'click' } when user clicks the notification body", async () => {
    scenarioOnNotificationCenter({
      callback: (cb) => cb(null, "activate", { activationType: "contentsClicked" }),
    })

    const result = await sendNotification({ title: "t", message: "m" })
    expect(result.type).toBe("click")
  })

  it("resolves with { type: 'error' } on notifier error", async () => {
    scenarioOnNotificationCenter({
      callback: (cb) => cb(new Error("broke"), ""),
    })

    const result = await sendNotification({ title: "t", message: "m" })
    expect(result.type).toBe("error")
    if (result.type === "error") {
      expect(result.error.message).toBe("broke")
    }
  })

  it("passes actions, closeLabel, timeout and sound through to node-notifier", async () => {
    const seen = scenarioOnNotificationCenter({
      callback: (cb) => cb(null, "timeout", { activationType: "timeout" }),
    })

    await sendNotification({
      title: "t",
      message: "m",
      actions: ["A", "B"],
      closeLabel: "Dismiss",
      timeoutSec: 42,
      sound: false,
    })

    expect(seen.length).toBe(1)
    expect(seen[0]?.actions).toEqual(["A", "B"])
    expect(seen[0]?.closeLabel).toBe("Dismiss")
    expect(seen[0]?.timeout).toBe(42)
    expect(seen[0]?.sound).toBe(false)
    expect(seen[0]?.wait).toBe(true)
  })

  it("defaults sound to true when not specified", async () => {
    const seen = scenarioOnNotificationCenter({
      callback: (cb) => cb(null, "timeout", { activationType: "timeout" }),
    })

    await sendNotification({ title: "t", message: "m" })
    expect(seen[0]?.sound).toBe(true)
  })
})

describe("sendNotification — non-macOS backend selection", () => {
  it("uses the cross-platform entry point, not NotificationCenter", async () => {
    const seen = scenarioOnCrossPlatform({
      callback: (cb) => cb(null, "read-only notification"),
    })

    const result = await sendNotification({
      title: "t",
      message: "m",
      actions: ["Approve", "Reject"],
    })

    expect(crossPlatformEntry.notify).toHaveBeenCalledTimes(1)
    // The regression this guards: constructing NotificationCenter on Linux
    // fails with "You need Mac OS X 10.8 or above…", so every notification
    // silently became an error and the user was never told anything.
    expect(NotificationCenter).not.toHaveBeenCalled()
    expect(seen[0]?.title).toBe("t")
    expect(seen[0]?.message).toBe("m")
    expect(result.type).toBe("timeout")
  })

  it("treats a plain notify-send success as informational (no buttons available)", async () => {
    // notify-send reports no activationType, so the only safe reading is
    // "the user saw it and did not choose anything": the TUI prompt decides.
    scenarioOnCrossPlatform({ callback: (cb) => cb(null, "") })

    const result = await sendNotification({
      title: "t",
      message: "m",
      actions: ["Approve", "Reject"],
    })

    expect(result.type).toBe("timeout")
  })

  it("still surfaces a notifier error", async () => {
    scenarioOnCrossPlatform({
      callback: (cb) => cb(new Error("notify-send missing"), ""),
    })

    const result = await sendNotification({ title: "t", message: "m" })
    expect(result).toEqual<NotifyActionResult>({
      type: "error",
      error: expect.any(Error),
    })
  })

  it("does not treat notify-send stderr noise as a failure", async () => {
    // node-notifier passes stderr in the error slot; a displayed notification
    // that printed a warning must not be reported as failed.
    scenarioOnCrossPlatform({
      callback: (cb) =>
        cb("Gtk-WARNING: cannot open display: " as never, ""),
    })

    const result = await sendNotification({ title: "t", message: "m" })
    expect(result.type).toBe("timeout")
  })
})
