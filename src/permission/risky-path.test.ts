import { describe, it, expect, vi, beforeEach } from "vitest"

vi.mock("../notify/notify.ts", () => ({
  sendNotification: vi.fn(),
}))

import { sendNotification } from "../notify/notify.ts"
import type { NotifyActionResult } from "../notify/notify.ts"
import { runRiskyPathInBackground } from "./risky-path.ts"
import { makeLogger } from "../testing/v2-fixtures.ts"

const mockedSend = vi.mocked(sendNotification)

beforeEach(() => {
  mockedSend.mockReset()
})

/** V2 injects a replier instead of exposing the SDK client. */
function makeReply(impl?: () => Promise<void>) {
  return vi.fn(impl ?? (async () => {}))
}

const baseArgs = {
  command: "rm -rf build",
  reason: "destructive rm",
  sound: true,
  timeoutSec: 60,
}

describe("runRiskyPathInBackground", () => {
  it("resolves the permission with 'once' when the user clicks Approve", async () => {
    mockedSend.mockResolvedValueOnce({
      type: "action",
      label: "Approve",
    } as NotifyActionResult)
    const reply = makeReply()

    await runRiskyPathInBackground({ ...baseArgs, reply })

    expect(reply).toHaveBeenCalledTimes(1)
    expect(reply).toHaveBeenCalledWith("once")
  })

  it("resolves the permission with 'reject' when the user clicks Reject", async () => {
    mockedSend.mockResolvedValueOnce({
      type: "action",
      label: "Reject",
    } as NotifyActionResult)
    const reply = makeReply()

    await runRiskyPathInBackground({ ...baseArgs, reply })

    expect(reply).toHaveBeenCalledTimes(1)
    expect(reply).toHaveBeenCalledWith("reject")
  })

  it("does NOT reply when the notification times out (user will decide in TUI)", async () => {
    mockedSend.mockResolvedValueOnce({ type: "timeout" } as NotifyActionResult)
    const reply = makeReply()

    await runRiskyPathInBackground({ ...baseArgs, reply })
    expect(reply).not.toHaveBeenCalled()
  })

  it("does NOT reply when the user dismisses the notification", async () => {
    mockedSend.mockResolvedValueOnce({ type: "cancel" } as NotifyActionResult)
    const reply = makeReply()

    await runRiskyPathInBackground({ ...baseArgs, reply })
    expect(reply).not.toHaveBeenCalled()
  })

  it("does NOT reply when the user clicks the notification body", async () => {
    mockedSend.mockResolvedValueOnce({ type: "click" } as NotifyActionResult)
    const reply = makeReply()

    await runRiskyPathInBackground({ ...baseArgs, reply })
    expect(reply).not.toHaveBeenCalled()
  })

  it("does NOT reply when the notifier errors (TUI is still available)", async () => {
    mockedSend.mockResolvedValueOnce({
      type: "error",
      error: new Error("no display"),
    } as NotifyActionResult)
    const reply = makeReply()

    await runRiskyPathInBackground({ ...baseArgs, reply })
    expect(reply).not.toHaveBeenCalled()
  })

  it("does NOT throw if the reply itself errors (TUI is still there to fall back on)", async () => {
    mockedSend.mockResolvedValueOnce({
      type: "action",
      label: "Approve",
    } as NotifyActionResult)
    const reply = makeReply(async () => {
      throw new Error("reply boom")
    })

    await expect(
      runRiskyPathInBackground({ ...baseArgs, reply }),
    ).resolves.toBeUndefined()
  })

  it("does NOT throw on unexpected action label — just leaves it for the TUI", async () => {
    mockedSend.mockResolvedValueOnce({
      type: "action",
      label: "Snooze",
    } as NotifyActionResult)
    const reply = makeReply()

    await runRiskyPathInBackground({ ...baseArgs, reply })
    // No reply for unknown actions; TUI prompt remains live.
    expect(reply).not.toHaveBeenCalled()
  })

  it("records a notifier error as a warning, since that silently drops the notification affordance", async () => {
    mockedSend.mockResolvedValueOnce({
      type: "error",
      error: new Error("no notification daemon"),
    } as NotifyActionResult)
    const reply = makeReply()
    const log = makeLogger()

    await runRiskyPathInBackground({ ...baseArgs, reply, log })

    expect(reply).not.toHaveBeenCalled()
    expect(log.warn).toHaveBeenCalledWith(
      "risky notification failed; TUI prompt remains",
      { error: "no notification daemon" },
    )
    expect(log.debug).not.toHaveBeenCalled()
  })

  it("records the notification outcome so the notifier is observable", async () => {
    mockedSend.mockResolvedValueOnce({
      type: "action",
      label: "Approve",
    } as NotifyActionResult)
    const reply = makeReply()
    const log = makeLogger()

    await runRiskyPathInBackground({ ...baseArgs, reply, log })

    expect(log.debug).toHaveBeenCalledWith("risky notification outcome", {
      outcome: "action",
      label: "Approve",
    })
  })

  it("passes Approve + Reject as action buttons, command + reason as context", async () => {
    mockedSend.mockResolvedValueOnce({ type: "timeout" } as NotifyActionResult)
    const reply = makeReply()

    await runRiskyPathInBackground({ ...baseArgs, reply })

    const args = mockedSend.mock.calls[0]?.[0]
    expect(args?.actions).toEqual(["Approve", "Reject"])
    expect(args?.message).toContain("rm -rf build")
    expect(args?.sound).toBe(true)
    expect(args?.timeoutSec).toBe(60)
    expect(args?.title.toLowerCase()).toMatch(/risky|review/)
  })

  it("truncates excessively long commands in the notification body", async () => {
    mockedSend.mockResolvedValueOnce({ type: "timeout" } as NotifyActionResult)
    const reply = makeReply()

    const longCmd = "curl " + "x".repeat(500)
    await runRiskyPathInBackground({
      ...baseArgs,
      command: longCmd,
      reply,
    })

    const args = mockedSend.mock.calls[0]?.[0]
    expect(args?.message.length).toBeLessThan(400)
  })
})
