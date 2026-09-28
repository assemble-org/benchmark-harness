import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { failureOption } from "effect/Cause";
import { flatMap, gen, provide, runPromise, runPromiseExit } from "effect/Effect";
import { getOrThrow } from "effect/Option";

import { assertFailure, assertSuccess } from "../../test/helpers/exit-asserts";
import { resetGenerationIds } from "../runtime/generation-ids";
import {
  accumulatorToResult,
  makeChatCompletionsLayer,
  responsesInputToChatMessages,
  responsesRequestToChat,
} from "./chat-completions-client";
import { Responses, ResponsesError } from "./responses-client";

describe("responsesInputToChatMessages", () => {
  it("adds a leading system message for instructions", () => {
    expect(responsesInputToChatMessages("hi", "be brief")).toEqual([
      { role: "system", content: "be brief" },
      { role: "user", content: "hi" },
    ]);
  });
  it("adds nothing for undefined or empty instructions", () => {
    expect(responsesInputToChatMessages("hi", undefined)).toEqual([
      { role: "user", content: "hi" },
    ]);
    expect(responsesInputToChatMessages("hi", "")).toEqual([
      { role: "user", content: "hi" },
    ]);
  });
  it("turns a bare string input into one user message", () => {
    expect(responsesInputToChatMessages("solve this", undefined)).toEqual([
      { role: "user", content: "solve this" },
    ]);
  });
  it("passes string content through and maps developer to system", () => {
    expect(
      responsesInputToChatMessages(
        [
          { type: "message", role: "developer", content: "rules" },
          { type: "message", role: "user", content: "question" },
        ],
        undefined
      )
    ).toEqual([
      { role: "system", content: "rules" },
      { role: "user", content: "question" },
    ]);
  });
  it("maps Responses content parts to chat content parts", () => {
    expect(
      responsesInputToChatMessages(
        [
          {
            type: "message",
            role: "user",
            content: [
              { type: "input_text", text: "a" },
              {
                type: "input_image",
                imageUrl: "http://x/i.png",
                detail: "low",
              },
            ],
          },
        ],
        undefined
      )
    ).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "a" },
          {
            type: "image_url",
            image_url: { url: "http://x/i.png", detail: "low" },
          },
        ],
      },
    ]);
  });
  it("joins text-only assistant content into a single string", () => {
    expect(
      responsesInputToChatMessages(
        [
          {
            type: "message",
            role: "assistant",
            content: [
              { type: "output_text", text: "Hello " },
              { type: "output_text", text: "world" },
            ],
          },
        ],
        undefined
      )
    ).toEqual([{ role: "assistant", content: "Hello world" }]);
  });
  it("attaches consecutive function_call items to the preceding assistant message", () => {
    const messages = responsesInputToChatMessages(
      [
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "working" }],
        },
        {
          type: "function_call",
          callId: "call-1",
          name: "bash",
          arguments: '{"command":"pwd"}',
        },
        {
          type: "function_call",
          callId: "call-2",
          name: "lookup",
          arguments: '{"x":1}',
        },
      ],
      undefined
    );
    expect(messages).toEqual([
      {
        role: "assistant",
        content: "working",
        tool_calls: [
          {
            id: "call-1",
            type: "function",
            function: { name: "bash", arguments: '{"command":"pwd"}' },
          },
          {
            id: "call-2",
            type: "function",
            function: { name: "lookup", arguments: '{"x":1}' },
          },
        ],
      },
    ]);
  });
  it("creates a fresh assistant message for a function_call with no preceding assistant message", () => {
    expect(
      responsesInputToChatMessages(
        [
          { type: "message", role: "user", content: "go" },
          {
            type: "function_call",
            callId: "call-1",
            name: "bash",
            arguments: "{}",
          },
        ],
        undefined
      )
    ).toEqual([
      { role: "user", content: "go" },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "call-1",
            type: "function",
            function: { name: "bash", arguments: "{}" },
          },
        ],
      },
    ]);
  });
  it("starts a new assistant message after a tool message rather than attaching to it", () => {
    const messages = responsesInputToChatMessages(
      [
        {
          type: "function_call_output",
          callId: "call-1",
          output: "ok",
        },
        {
          type: "function_call",
          callId: "call-2",
          name: "bash",
          arguments: "{}",
        },
      ],
      undefined
    );
    expect(messages).toEqual([
      { role: "tool", tool_call_id: "call-1", content: "ok" },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "call-2",
            type: "function",
            function: { name: "bash", arguments: "{}" },
          },
        ],
      },
    ]);
  });
  it("maps function_call_output to a tool message and JSON-stringifies non-string output", () => {
    expect(
      responsesInputToChatMessages(
        [
          { type: "function_call_output", callId: "c1", output: "plain" },
          { type: "function_call_output", call_id: "c2", output: { a: 1 } },
        ],
        undefined
      )
    ).toEqual([
      { role: "tool", tool_call_id: "c1", content: "plain" },
      { role: "tool", tool_call_id: "c2", content: '{"a":1}' },
    ]);
  });
  it("drops reasoning items", () => {
    expect(
      responsesInputToChatMessages(
        [
          { type: "reasoning", id: "rs_1", summary: [] },
          { type: "message", role: "user", content: "hi" },
        ],
        undefined
      )
    ).toEqual([{ role: "user", content: "hi" }]);
  });
});

describe("responsesRequestToChat", () => {
  it("builds the chat-completions request body", () => {
    const body = responsesRequestToChat(
      {
        model: "openai/gpt-5",
        input: [{ type: "message", role: "user", content: "hi" }],
        instructions: "Use bash.",
        temperature: 0,
        maxOutputTokens: 256,
        reasoning: { effort: "high" },
        tools: [
          {
            type: "function",
            name: "bash",
            parameters: { type: "object" },
            description: "Run bash.",
          },
        ],
      } as never,
      undefined
    );
    expect(body).toMatchObject({
      model: "openai/gpt-5",
      messages: [
        { role: "system", content: "Use bash." },
        { role: "user", content: "hi" },
      ],
      stream: true,
      stream_options: { include_usage: true },
      temperature: 0,
      max_tokens: 256,
      reasoning_effort: "high",
      tools: [
        {
          type: "function",
          function: {
            name: "bash",
            parameters: { type: "object" },
            description: "Run bash.",
          },
        },
      ],
    });
  });
  it("merges extraBody keys over the built body", () => {
    const body = responsesRequestToChat(
      { model: "m", input: "hi" } as never,
      { top_k: 5 }
    );
    expect(body).toMatchObject({ top_k: 5 });
    const overridden = responsesRequestToChat(
      { model: "m", input: "hi", maxOutputTokens: 256 } as never,
      { max_tokens: 7 }
    );
    expect(overridden).toMatchObject({ max_tokens: 7 });
  });
});

describe("accumulatorToResult", () => {
  function acc(overrides: Record<string, unknown> = {}): Record<string, unknown> {
    return {
      id: "chatcmpl-1",
      model: "openai/gpt-5",
      text: "",
      reasoning: "",
      toolCalls: new Map(),
      finishReason: null,
      usage: null,
      sawDone: true,
      ...overrides,
    };
  }
  it("emits reasoning first, then the message, then tool calls in index order", () => {
    const result = accumulatorToResult(
      acc({
        text: "done",
        reasoning: "thinking",
        toolCalls: new Map([
          [1, { id: "call-2", name: "lookup", arguments: '{"y":2}' }],
          [0, { id: "call-1", name: "bash", arguments: '{"x":1}' }],
        ]),
      }) as never,
      10
    );
    expect(result.output).toEqual([
      {
        type: "reasoning",
        id: "chatcmpl-1-reasoning",
        summary: [],
        content: [{ type: "reasoning_text", text: "thinking" }],
      },
      {
        type: "message",
        id: "chatcmpl-1-message",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "done", annotations: [] }],
      },
      {
        type: "function_call",
        id: "chatcmpl-1-call-1",
        callId: "call-1",
        name: "bash",
        arguments: '{"x":1}',
        status: "completed",
      },
      {
        type: "function_call",
        id: "chatcmpl-1-call-2",
        callId: "call-2",
        name: "lookup",
        arguments: '{"y":2}',
        status: "completed",
      },
    ]);
    expect(result.text).toBe("done");
    expect(result.status).toBe("completed");
    expect(result.id).toBe("chatcmpl-1");
    expect(result.generationId).toBe("chatcmpl-1");
  });
  it("marks length finish as incomplete", () => {
    const result = accumulatorToResult(
      acc({ text: "partial", finishReason: "length" }) as never,
      10
    );
    expect(result.status).toBe("incomplete");
  });
  it("maps chat usage to Responses usage", () => {
    const result = accumulatorToResult(
      acc({
        usage: {
          prompt_tokens: 3,
          completion_tokens: 5,
          total_tokens: 8,
          completion_tokens_details: { reasoning_tokens: 2 },
        },
      }) as never,
      10
    );
    expect(result.usage).toEqual({
      inputTokens: 3,
      outputTokens: 5,
      totalTokens: 8,
      outputTokensDetails: { reasoningTokens: 2 },
    });
  });
  it("still yields one empty message item for an empty accumulator", () => {
    const result = accumulatorToResult(acc() as never, 10);
    expect(result.output).toEqual([
      {
        type: "message",
        id: "chatcmpl-1-message",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "", annotations: [] }],
      },
    ]);
    expect(result.text).toBe("");
  });
});

describe("makeChatCompletionsLayer", () => {
  let originalFetch: typeof globalThis.fetch;
  let capturedRequest: Request | undefined;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    capturedRequest = undefined;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  function installFetchStub(
    responseBody: string,
    status: number,
    responseHeaders: Record<string, string> = {}
  ): void {
    globalThis.fetch = async (input, init) => {
      const request = input instanceof Request ? input : new Request(input, init);
      capturedRequest = request;
      return new Response(responseBody, {
        status,
        headers: {
          "content-type": "text/event-stream",
          ...responseHeaders,
        },
      });
    };
  }

  function sseResponse(events: unknown[]): string {
    return events
      .map(
        (event) =>
          `data: ${typeof event === "string" ? event : JSON.stringify(event)}\n\n`
      )
      .join("");
  }

  function streamingBody(): string {
    return (
      ": keep-alive\n\n" +
      sseResponse([
        { id: "chatcmpl-1", model: "openai/gpt-5", choices: [] },
        {
          id: "chatcmpl-1",
          model: "openai/gpt-5",
          choices: [
            { delta: { content: "Hello " }, index: 0 },
          ],
        },
        {
          id: "chatcmpl-1",
          choices: [
            { delta: { reasoning_content: "thinking" }, index: 0 },
          ],
        },
        {
          id: "chatcmpl-1",
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  { index: 0, id: "call-1", function: { name: "bash", arguments: '{"comm' } },
                ],
              },
            },
          ],
        },
        {
          id: "chatcmpl-1",
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  { index: 0, function: { arguments: 'and":"pwd"}' } },
                ],
              },
            },
          ],
        },
        {
          id: "chatcmpl-1",
          choices: [
            { index: 0, delta: { content: "world" }, finish_reason: "tool_calls" },
          ],
        },
        {
          id: "chatcmpl-1",
          choices: [],
          usage: {
            prompt_tokens: 3,
            completion_tokens: 5,
            total_tokens: 8,
            completion_tokens_details: { reasoning_tokens: 2 },
          },
        },
        "[DONE]",
      ])
    );
  }

  it("streams a chat completion end to end", async () => {
    installFetchStub(streamingBody(), 200);
    const exit = await runPromiseExit(
      resetGenerationIds.pipe(
        flatMap(() =>
          gen(function* run() {
            const responses = yield* Responses;
            return yield* responses.send(
              { model: "openai/gpt-5", input: "hi" } as never,
              { timeoutMs: 1000 }
            );
          })
        ),
        provide(
          makeChatCompletionsLayer({
            model: "openai/gpt-5",
            apiKey: "sk-test",
            baseUrl: "https://example.test/openrouter/v1",
          })
        )
      )
    );
    assertSuccess(exit);
    expect(capturedRequest?.url).toBe(
      "https://example.test/openrouter/v1/chat/completions"
    );
    expect(capturedRequest?.headers.get("authorization")).toBe("Bearer sk-test");
    const body: unknown = JSON.parse(await capturedRequest!.clone().text());
    expect(body).toMatchObject({ stream: true });
    expect(exit.value.text).toBe("Hello world");
    expect(exit.value.output[0]).toEqual({
      type: "reasoning",
      id: "chatcmpl-1-reasoning",
      summary: [],
      content: [{ type: "reasoning_text", text: "thinking" }],
    });
    expect(exit.value.output).toContainEqual({
      type: "function_call",
      id: "chatcmpl-1-call-1",
      callId: "call-1",
      name: "bash",
      arguments: '{"command":"pwd"}',
      status: "completed",
    });
    expect(exit.value.usage).toEqual({
      inputTokens: 3,
      outputTokens: 5,
      totalTokens: 8,
      outputTokensDetails: { reasoningTokens: 2 },
    });
  });

  it("appends /api/v1 for the openrouter.ai host", async () => {
    installFetchStub(streamingBody(), 200);
    const exit = await runPromiseExit(
      gen(function* run() {
        const responses = yield* Responses;
        return yield* responses.send({ model: "m", input: [] } as never, {
          timeoutMs: 1000,
        });
      }).pipe(
        provide(
          makeChatCompletionsLayer({
            model: "m",
            apiKey: "sk-test",
            baseUrl: "https://openrouter.ai",
          })
        )
      )
    );
    assertSuccess(exit);
    expect(capturedRequest?.url).toBe(
      "https://openrouter.ai/api/v1/chat/completions"
    );
  });

  it("maps a 429 with retry-after to a retryable ResponsesError", async () => {
    installFetchStub("rate limited", 429, { "retry-after": "2" });
    const exit = await runPromiseExit(
      gen(function* run() {
        const responses = yield* Responses;
        return yield* responses.send({ model: "m", input: [] } as never, {
          timeoutMs: 1000,
        });
      }).pipe(
        provide(
          makeChatCompletionsLayer({ model: "m", apiKey: "sk-test" })
        )
      )
    );
    assertFailure(exit);
    const error = getOrThrow(failureOption(exit.cause));
    expect(error).toBeInstanceOf(ResponsesError);
    expect(error.status).toBe(429);
    expect(error.retryable).toBe(true);
    expect(error.retryAfterMs).toBe(2000);
  });

  it("fails retryably on an in-stream error payload", async () => {
    installFetchStub(
      sseResponse([{ error: { code: 502, message: "upstream died" } }]),
      200
    );
    const exit = await runPromiseExit(
      gen(function* run() {
        const responses = yield* Responses;
        return yield* responses.send({ model: "m", input: [] } as never, {
          timeoutMs: 1000,
        });
      }).pipe(
        provide(
          makeChatCompletionsLayer({ model: "m", apiKey: "sk-test" })
        )
      )
    );
    assertFailure(exit);
    const error = getOrThrow(failureOption(exit.cause));
    expect(error).toBeInstanceOf(ResponsesError);
    expect(error.status).toBe(502);
    expect(error.retryable).toBe(true);
  });

  it("fails retryably when the stream ends without [DONE] or a finish_reason", async () => {
    installFetchStub(
      sseResponse([
        { id: "chatcmpl-1", choices: [{ index: 0, delta: { content: "partial" } }] },
      ]),
      200
    );
    const exit = await runPromiseExit(
      gen(function* run() {
        const responses = yield* Responses;
        return yield* responses.send({ model: "m", input: [] } as never, {
          timeoutMs: 1000,
        });
      }).pipe(
        provide(
          makeChatCompletionsLayer({ model: "m", apiKey: "sk-test" })
        )
      )
    );
    assertFailure(exit);
    const error = getOrThrow(failureOption(exit.cause));
    expect(error).toBeInstanceOf(ResponsesError);
    expect(error.retryable).toBe(true);
    expect(error.message).toContain(
      "Stream ended without [DONE] or a finish_reason"
    );
  });
});

describe("makeChatCompletionsLayer gateway deadline", () => {
  let originalFetch: typeof globalThis.fetch;
  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("maps an OpenAI-shaped deadline frame with no numeric code to 504", async () => {
    globalThis.fetch = async () =>
      new Response(
        'data: {"error":{"message":"The upstream connection went silent past the gateway\'s limit. Retry the request to continue on another backend.","type":"server_error","param":null,"code":null}}\n\ndata: [DONE]\n\n',
        { status: 200, headers: { "content-type": "text/event-stream" } }
      );
    const exit = await runPromiseExit(
      gen(function* run() {
        const responses = yield* Responses;
        return yield* responses.send({ model: "m", input: [] } as never, {
          timeoutMs: 1000,
        });
      }).pipe(provide(makeChatCompletionsLayer({ apiKey: "sk-test" })))
    );
    assertFailure(exit);
    const error = getOrThrow(failureOption(exit.cause));
    expect(error).toBeInstanceOf(ResponsesError);
    expect(error.status).toBe(504);
    expect(error.retryable).toBe(true);
  });
});

describe("chat wire without streaming", () => {
  let originalFetch: typeof globalThis.fetch;
  let captured: Request | undefined;
  beforeEach(() => {
    originalFetch = globalThis.fetch;
    captured = undefined;
    process.env["OPENROUTER_STREAM"] = "false";
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
    delete process.env["OPENROUTER_STREAM"];
  });

  function stub(body: string, status = 200): void {
    globalThis.fetch = async (input, init) => {
      captured = input instanceof Request ? input : new Request(input, init);
      return new Response(body, { status, headers: { "content-type": "application/json" } });
    };
  }

  async function send() {
    return runPromiseExit(
      gen(function* run() {
        const responses = yield* Responses;
        return yield* responses.send({ model: "m", input: "hi" } as never, { timeoutMs: 1000 });
      }).pipe(provide(makeChatCompletionsLayer({ apiKey: "sk-test", baseUrl: "https://example.test/v1" })))
    );
  }

  it("sends stream:false and reads one body, after a whitespace heartbeat", async () => {
    stub(
      "\n\n   " +
        JSON.stringify({
          id: "chatcmpl-9",
          model: "m",
          choices: [
            {
              index: 0,
              finish_reason: "stop",
              message: { role: "assistant", content: "Answer: B", reasoning: "think" },
            },
          ],
          usage: { prompt_tokens: 4, completion_tokens: 6, total_tokens: 10 },
        })
    );
    const exit = await send();
    assertSuccess(exit);
    const sent: Record<string, unknown> = JSON.parse(await captured!.clone().text());
    expect(sent["stream"]).toBe(false);
    expect(sent["stream_options"]).toBeUndefined();
    expect(exit.value.text).toBe("Answer: B");
    expect(exit.value.output[0]).toMatchObject({ type: "reasoning" });
    expect(exit.value.usage).toMatchObject({ outputTokens: 6 });
  });

  it("maps an error-only 200 body to a ResponsesError", async () => {
    stub(
      JSON.stringify({
        error: {
          message: "The upstream connection went silent past the gateway's limit.",
          type: "server_error",
          param: null,
          code: null,
        },
      })
    );
    const exit = await send();
    assertFailure(exit);
    const error = getOrThrow(failureOption(exit.cause));
    expect(error.status).toBe(504);
    expect(error.retryable).toBe(true);
  });
});
