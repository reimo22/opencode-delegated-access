import { describe, it, expect, vi } from "vitest"
import {
  extractLastUserMessages,
  extractLatestAssistantModel,
  getSessionMessages,
  sanitizeUserMessageForClassifier,
  CLASSIFIER_MESSAGE_MAX_CHARS,
} from "./messages.ts"
import type { SessionContextReader, TranscriptMessage } from "./messages.ts"
import { assistantMessage, userMessage } from "../testing/v2-fixtures.ts"

describe("extractLastUserMessages", () => {
  it("returns an empty array when K is 0", () => {
    const entries = [userMessage("hello"), userMessage("world")]
    expect(extractLastUserMessages(entries, 0)).toEqual([])
  })

  it("returns the last K user messages in chronological order", () => {
    const entries = [
      userMessage("first"),
      userMessage("second"),
      userMessage("third"),
      userMessage("fourth"),
    ]
    expect(extractLastUserMessages(entries, 2)).toEqual(["third", "fourth"])
  })

  it("ignores assistant messages", () => {
    const entries = [
      userMessage("user first"),
      assistantMessage(undefined, "assistant reply"),
      userMessage("user second"),
      assistantMessage(undefined, "another assistant reply"),
    ]
    expect(extractLastUserMessages(entries, 2)).toEqual([
      "user first",
      "user second",
    ])
  })

  it("returns all user messages when K exceeds the count", () => {
    const entries = [userMessage("alpha"), userMessage("beta")]
    expect(extractLastUserMessages(entries, 10)).toEqual(["alpha", "beta"])
  })

  it("returns an empty array when there are no user messages", () => {
    expect(
      extractLastUserMessages([assistantMessage(undefined, "hi")], 3),
    ).toEqual([])
  })

  it("returns an empty array for an empty input", () => {
    expect(extractLastUserMessages([], 3)).toEqual([])
  })

  it("sanitizes injected blocks out of each returned message", () => {
    const entries = [
      userMessage("<pr_context>huge</pr_context>do the rename"),
      userMessage("plain request"),
    ]
    expect(extractLastUserMessages(entries, 2)).toEqual([
      "do the rename",
      "plain request",
    ])
  })

  it("drops a user message that becomes empty after sanitization", () => {
    // A pure pr_context injection contributes nothing once stripped, so it
    // must not appear as an empty string (and must not consume a K slot).
    const entries = [
      userMessage("<pr_context>only injected content</pr_context>"),
      userMessage("real one"),
      userMessage("real two"),
    ]
    expect(extractLastUserMessages(entries, 2)).toEqual(["real one", "real two"])
  })

  it("skips a user message with no text content", () => {
    // V2 transcript messages are flat: text lives inline, so a user turn with
    // no string `text` (e.g. an image-only message) contributes nothing.
    const entry: TranscriptMessage = { type: "user" }
    expect(extractLastUserMessages([entry], 1)).toEqual([])
  })
})

describe("sanitizeUserMessageForClassifier", () => {
  it("returns clean prose unchanged", () => {
    const text = "please refactor the auth module and run the tests"
    expect(sanitizeUserMessageForClassifier(text)).toBe(text)
  })

  it("strips a premind <pr_context> block but keeps surrounding prose", () => {
    const text =
      "reply on the PR\n<pr_context>\n<pr_meta>{...huge json...}</pr_meta>\n</pr_context>\nthanks"
    const out = sanitizeUserMessageForClassifier(text)
    expect(out).not.toMatch(/<pr_context>/)
    expect(out).not.toMatch(/pr_meta/)
    expect(out).toContain("reply on the PR")
    expect(out).toContain("thanks")
  })

  it("strips a superpowers <EXTREMELY_IMPORTANT> preamble", () => {
    const text =
      "<EXTREMELY_IMPORTANT>\nYou have superpowers. You MUST invoke skills...\n</EXTREMELY_IMPORTANT>\nmake the decider more lenient"
    const out = sanitizeUserMessageForClassifier(text)
    expect(out).not.toMatch(/EXTREMELY_IMPORTANT/)
    expect(out).not.toMatch(/superpowers/)
    expect(out).toContain("make the decider more lenient")
  })

  it("strips the hyphenated <EXTREMELY-IMPORTANT> variant", () => {
    const text =
      "<EXTREMELY-IMPORTANT>\nrules here\n</EXTREMELY-IMPORTANT>\nactual request"
    const out = sanitizeUserMessageForClassifier(text)
    expect(out).not.toMatch(/EXTREMELY-IMPORTANT/)
    expect(out).toContain("actual request")
  })

  it("strips <system-reminder> blocks", () => {
    const text =
      "do the thing\n<system-reminder>\nMode changed to build.\n</system-reminder>"
    const out = sanitizeUserMessageForClassifier(text)
    expect(out).not.toMatch(/system-reminder/)
    expect(out).not.toMatch(/Mode changed/)
    expect(out).toContain("do the thing")
  })

  it("strips <SUBAGENT-STOP> and <available_skills> blocks", () => {
    const text =
      "<SUBAGENT-STOP>\nskip\n</SUBAGENT-STOP>\nreal\n<available_skills>\n<skill>x</skill>\n</available_skills>"
    const out = sanitizeUserMessageForClassifier(text)
    expect(out).not.toMatch(/SUBAGENT-STOP/)
    expect(out).not.toMatch(/available_skills/)
    expect(out).toContain("real")
  })

  it("strips multiple blocks of different kinds in one message", () => {
    const text =
      "<EXTREMELY_IMPORTANT>a</EXTREMELY_IMPORTANT>command me<pr_context>b</pr_context>"
    const out = sanitizeUserMessageForClassifier(text)
    expect(out).not.toMatch(/EXTREMELY_IMPORTANT|pr_context/)
    expect(out).toContain("command me")
  })

  it("strips an UNCLOSED block to end-of-string (truncated injection)", () => {
    const text = "real request\n<pr_context>\n<pr_meta>{ huge json that got cut off"
    const out = sanitizeUserMessageForClassifier(text)
    expect(out).not.toMatch(/pr_context|pr_meta/)
    expect(out).toContain("real request")
  })

  it("is case-insensitive on tag names", () => {
    const text = "<Pr_Context>x</Pr_Context>keep me"
    const out = sanitizeUserMessageForClassifier(text)
    expect(out.toLowerCase()).not.toMatch(/pr_context/)
    expect(out).toContain("keep me")
  })

  it("returns an empty string when the message is ONLY an injected block", () => {
    const text = "<pr_context>\n<pr_meta>{...}</pr_meta>\n</pr_context>"
    expect(sanitizeUserMessageForClassifier(text)).toBe("")
  })

  it("caps an over-long message at CLASSIFIER_MESSAGE_MAX_CHARS", () => {
    const long = "x".repeat(CLASSIFIER_MESSAGE_MAX_CHARS + 5000)
    const out = sanitizeUserMessageForClassifier(long)
    expect(out.length).toBeLessThanOrEqual(CLASSIFIER_MESSAGE_MAX_CHARS)
  })

  it("does not cap a message at or below the limit", () => {
    const ok = "y".repeat(CLASSIFIER_MESSAGE_MAX_CHARS - 100)
    expect(sanitizeUserMessageForClassifier(ok)).toBe(ok)
  })

  it("applies the cap AFTER stripping (a huge pr_context doesn't consume the budget)", () => {
    // A short real request followed by a massive pr_context block. After
    // stripping, the result is well under the cap and fully preserved.
    const realRequest = "rename the helper and commit"
    const text = realRequest + "<pr_context>" + "j".repeat(500_000) + "</pr_context>"
    const out = sanitizeUserMessageForClassifier(text)
    expect(out).toBe(realRequest)
  })
})

describe("getSessionMessages", () => {
  it("calls session.context with the session id and returns the raw messages", async () => {
    const entries = [userMessage("alpha"), assistantMessage(undefined, "reply")]

    let receivedID: string | undefined
    const session: SessionContextReader = {
      context: vi.fn(async ({ sessionID }) => {
        receivedID = sessionID
        return entries
      }),
    }

    const result = await getSessionMessages(session, "sess_abc")
    expect(receivedID).toBe("sess_abc")
    expect(result).toEqual(entries)
  })

  it("returns an empty array when the context result is undefined", async () => {
    const session: SessionContextReader = {
      context: vi.fn(async () => undefined),
    }
    await expect(getSessionMessages(session, "sess_x")).resolves.toEqual([])
  })

  it("returns an empty array when the context result is null", async () => {
    const session: SessionContextReader = {
      context: vi.fn(async () => null as unknown as TranscriptMessage[]),
    }
    await expect(getSessionMessages(session, "sess_x")).resolves.toEqual([])
  })

  it("feeds the fetched messages to extractLastUserMessages", async () => {
    const messages = [
      userMessage("alpha"),
      userMessage("beta"),
      userMessage("gamma"),
    ]
    const session: SessionContextReader = {
      context: vi.fn(async ({ sessionID }) => {
        void sessionID
        return messages
      }),
    }

    const fetched = await getSessionMessages(session, "sess_123")
    expect(extractLastUserMessages(fetched, 2)).toEqual(["beta", "gamma"])
  })
})

describe("extractLatestAssistantModel", () => {
  it("returns provider+model from the latest assistant message", () => {
    const entries = [
      userMessage("hi"),
      assistantMessage(
        { providerID: "anthropic", id: "claude-sonnet-4-5" },
        "hello",
      ),
      userMessage("thanks"),
    ]
    expect(extractLatestAssistantModel(entries)).toEqual({
      providerID: "anthropic",
      modelID: "claude-sonnet-4-5",
    })
  })

  it("returns the MOST RECENT assistant's model when multiple exist", () => {
    const older = assistantMessage(
      { providerID: "anthropic", id: "claude-sonnet-4-5" },
      "older",
    )
    const newer = assistantMessage(
      { providerID: "openai", id: "gpt-4.1-mini" },
      "newer",
    )
    const entries = [older, userMessage("more"), newer]
    expect(extractLatestAssistantModel(entries)).toEqual({
      providerID: "openai",
      modelID: "gpt-4.1-mini",
    })
  })

  it("returns null for an empty input", () => {
    expect(extractLatestAssistantModel([])).toBeNull()
  })

  it("returns null when there are no assistant messages", () => {
    const entries = [userMessage("a"), userMessage("b")]
    expect(extractLatestAssistantModel(entries)).toBeNull()
  })

  it("skips assistants with non-string model fields and keeps searching older entries", () => {
    const bad: TranscriptMessage = {
      type: "assistant",
      model: {
        providerID: 42, // not a string
        id: undefined,
      } as unknown as TranscriptMessage["model"],
    }
    const good = assistantMessage(
      { providerID: "anthropic", id: "claude-sonnet-4-5" },
      "ok",
    )
    // `good` comes before `bad` chronologically but `bad` is latest and must be
    // skipped; we then fall back to `good`.
    const entries = [good, bad]
    expect(extractLatestAssistantModel(entries)).toEqual({
      providerID: "anthropic",
      modelID: "claude-sonnet-4-5",
    })
  })

  it("returns null when every assistant has unusable model fields", () => {
    const bad: TranscriptMessage = { type: "assistant" }
    expect(extractLatestAssistantModel([userMessage("x"), bad])).toBeNull()
  })
})
