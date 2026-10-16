// skypilot-patches: a chat-completions client behind the Responses service.
//
// The harness speaks the Responses API to OpenRouter. A provider being
// tested directly serves /chat/completions, so this layer implements the
// same `Responses.send` contract by translating the Responses request into
// a streaming chat-completions request and assembling the SSE chunks back
// into a ResponsesResult. Everything above `Responses` (generate, the
// benchmarks, scoring) is unchanged.
//
// Selected with OPENROUTER_WIRE=chat (the --wire CLI flag sets it).
//
// Limits: reasoning items in the input are dropped on replay (chat
// completions has no standard field for prior reasoning), and the
// cache-salt / response-cache headers are not sent, since they only mean
// something to OpenRouter's edge.
import type { ResponsesRequest } from "@openrouter/sdk/models";
import type { StreamEvents } from "@openrouter/sdk/models/streamevents";
import type { Effect } from "effect/Effect";
import { all, flatMap, map, tryPromise } from "effect/Effect";
import type { Layer } from "effect/Layer";
import { succeed as layerSucceed } from "effect/Layer";
import { definedValues, isRecord } from "../internal/guards";
import { filterTraceHeaders } from "../runner/trace-headers";
import { recordGenerationId } from "../runtime/generation-ids";
import {
  buildRequestSessionId,
  getCurrentSampleId,
} from "../runtime/request-session-id";
import {
  getCurrentEpoch,
  getCurrentRetryAttempt,
} from "../runtime/response-cache";
import { BENCH_HARNESS_APP_REFERRER, BENCH_HARNESS_APP_TITLE } from "./app-identity";
import type { ModelErrorIdentifiers } from "./request-identifiers";
import {
  appendModelErrorIdentifiers,
  modelErrorIdentifiersFromFetchHeaders,
} from "./request-identifiers";
import type {
  ResponsesConfig,
  ResponsesResult,
  ResponsesSendOptions,
} from "./responses-client";
import {
  Responses,
  ResponsesError,
  extractMessageText,
  parseRetryAfter,
  providerNameFromErrorBody,
} from "./responses-client";

export const WIRE_ENV = "OPENROUTER_WIRE";
// STREAM_ENV=false makes the chat wire send `stream: false` and read one
// JSON body. Only the chat wire honours it; the Responses client always
// streams. The --no-stream CLI flag sets it.
export const STREAM_ENV = "OPENROUTER_STREAM";

// chatStreamsFromEnv returns false only when OPENROUTER_STREAM is "false".
export function chatStreamsFromEnv(): boolean {
  return process.env[STREAM_ENV] !== "false";
}
export const WIRES = ["responses", "chat"] as const;
export type Wire = (typeof WIRES)[number];

// wireFromEnv returns the wire named by OPENROUTER_WIRE, defaulting to
// "responses". An unknown value is an error at the first request rather
// than a silent fallback.
export function wireFromEnv(): Wire {
  const raw = process.env[WIRE_ENV];
  if (raw === undefined || raw === "") {
    return "responses";
  }
  if (raw === "responses" || raw === "chat") {
    return raw;
  }
  throw new Error(`${WIRE_ENV} must be one of ${WIRES.join(", ")}, got "${raw}"`);
}

// chatBaseUrl mirrors normalizeBaseUrl in responses-client.ts: only
// openrouter.ai gets the implicit /api/v1 suffix.
function chatBaseUrl(baseUrl: string | undefined): string {
  const trimmed = (baseUrl ?? "https://openrouter.ai").replace(/\/+$/u, "");
  if (!/^https?:\/\/([^/]*\.)?openrouter\.ai(\/|$)/u.test(trimmed)) {
    return trimmed;
  }
  return trimmed.endsWith("/api/v1") ? trimmed : `${trimmed}/api/v1`;
}

type ChatMessage = Record<string, unknown>;

function contentPartToChat(part: unknown): unknown {
  if (!isRecord(part)) {
    return part;
  }
  switch (part["type"]) {
    case "input_text":
    case "output_text":
      return { type: "text", text: part["text"] };
    case "input_image":
      return {
        type: "image_url",
        image_url: definedValues({
          url: part["imageUrl"] ?? part["image_url"],
          detail: part["detail"],
        }),
      };
    default:
      return part;
  }
}

function messageContentToChat(content: unknown): unknown {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  const parts = content.map(contentPartToChat);
  // Text-only assistant content is sent as one string: every engine
  // accepts it and some reject part arrays on assistant turns.
  const allText = parts.every(
    (p) => isRecord(p) && p["type"] === "text" && typeof p["text"] === "string"
  );
  if (allText) {
    return parts.map((p) => (p as { text: string }).text).join("");
  }
  return parts;
}

// responsesInputToChatMessages turns the Responses input items (in the
// SDK's camelCase form, or a bare string) into chat messages. Consecutive
// function_call items attach to the assistant message before them, which
// is how chat completions expresses one turn with several tool calls.
export function responsesInputToChatMessages(
  input: unknown,
  instructions: string | undefined
): ChatMessage[] {
  const messages: ChatMessage[] = [];
  if (instructions !== undefined && instructions.length > 0) {
    messages.push({ role: "system", content: instructions });
  }
  if (typeof input === "string") {
    messages.push({ role: "user", content: input });
    return messages;
  }
  if (!Array.isArray(input)) {
    return messages;
  }
  for (const raw of input) {
    if (!isRecord(raw)) {
      continue;
    }
    const type = raw["type"] ?? "message";
    switch (type) {
      case "message": {
        const role = raw["role"] === "developer" ? "system" : raw["role"];
        messages.push({ role, content: messageContentToChat(raw["content"]) });
        break;
      }
      case "function_call": {
        const call = {
          id: raw["callId"] ?? raw["call_id"],
          type: "function",
          function: { name: raw["name"], arguments: raw["arguments"] },
        };
        const last = messages[messages.length - 1];
        if (
          last !== undefined &&
          last["role"] === "assistant" &&
          !("tool_call_id" in last)
        ) {
          const calls = Array.isArray(last["tool_calls"]) ? last["tool_calls"] : [];
          last["tool_calls"] = [...calls, call];
        } else {
          messages.push({ role: "assistant", content: "", tool_calls: [call] });
        }
        break;
      }
      case "function_call_output": {
        const output = raw["output"];
        messages.push({
          role: "tool",
          tool_call_id: raw["callId"] ?? raw["call_id"],
          content: typeof output === "string" ? output : JSON.stringify(output),
        });
        break;
      }
      case "reasoning":
        // Dropped on replay; see the file comment.
        break;
      default:
        break;
    }
  }
  return messages;
}

function toolToChat(tool: unknown): unknown {
  if (!isRecord(tool) || tool["type"] !== "function") {
    return tool;
  }
  return {
    type: "function",
    function: definedValues({
      name: tool["name"],
      parameters: tool["parameters"],
      description: tool["description"],
      strict: tool["strict"],
    }),
  };
}

// responsesRequestToChat builds the chat-completions request body. Caller
// extraBody (already snake_case) is merged over ours, matching the
// Responses layer's mergeExtraBody.
export function responsesRequestToChat(
  body: ResponsesRequest,
  extraBody: Readonly<Record<string, unknown>> | undefined,
  stream = true
): Record<string, unknown> {
  const b = body as unknown as Record<string, unknown>;
  const reasoning = isRecord(b["reasoning"]) ? b["reasoning"] : undefined;
  const tools = Array.isArray(b["tools"]) ? b["tools"].map(toolToChat) : undefined;
  return {
    model: b["model"],
    messages: responsesInputToChatMessages(
      b["input"],
      typeof b["instructions"] === "string" ? b["instructions"] : undefined
    ),
    stream,
    ...(stream ? { stream_options: { include_usage: true } } : {}),
    ...definedValues({
      temperature: b["temperature"],
      max_tokens: b["maxOutputTokens"] ?? b["max_output_tokens"],
      tools: tools !== undefined && tools.length > 0 ? tools : undefined,
      reasoning_effort: reasoning?.["effort"],
    }),
    ...extraBody,
  };
}

// nonStreamBodyToAccumulator reads one chat.completion object into the
// same accumulator the stream path fills. A body that carries an `error`
// object instead of choices (a gateway that failed after committing its
// 200) throws the same ResponsesError a stream error frame would. Leading
// whitespace, which a gateway may send as a heartbeat before the body, is
// valid JSON and needs no handling.
export function nonStreamBodyToAccumulator(
  text: string,
  identifiers: ModelErrorIdentifiers
): ChatAccumulator {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new ResponsesError({
      message: appendModelErrorIdentifiers(
        `Chat body was not JSON: ${text.trim().slice(0, 200)}`,
        identifiers
      ),
      status: 502,
      retryable: true,
      ...identifiers,
    });
  }
  if (!isRecord(parsed)) {
    throw new ResponsesError({ message: "Chat body was not a JSON object", status: 502, retryable: true, ...identifiers });
  }
  if (isRecord(parsed["error"])) {
    throw streamErrorToResponsesError(parsed["error"], identifiers);
  }
  const acc: ChatAccumulator = {
    id: typeof parsed["id"] === "string" ? parsed["id"] : null,
    model: typeof parsed["model"] === "string" ? parsed["model"] : null,
    text: "",
    reasoning: "",
    toolCalls: new Map(),
    finishReason: null,
    usage: isRecord(parsed["usage"]) ? parsed["usage"] : null,
    sawDone: true,
  };
  const choices = parsed["choices"];
  const choice = Array.isArray(choices) && isRecord(choices[0]) ? choices[0] : undefined;
  if (choice === undefined) {
    throw new ResponsesError({ message: appendModelErrorIdentifiers("Chat body had no choices", identifiers), status: 502, retryable: true, ...identifiers });
  }
  if (typeof choice["finish_reason"] === "string") {
    acc.finishReason = choice["finish_reason"];
  }
  const message = isRecord(choice["message"]) ? choice["message"] : {};
  if (typeof message["content"] === "string") {
    acc.text = message["content"];
  }
  const reasoning = message["reasoning"] ?? message["reasoning_content"];
  if (typeof reasoning === "string") {
    acc.reasoning = reasoning;
  }
  if (Array.isArray(message["tool_calls"])) {
    message["tool_calls"].forEach((call, index) => {
      if (!isRecord(call)) {
        return;
      }
      const fn = isRecord(call["function"]) ? call["function"] : {};
      acc.toolCalls.set(index, {
        id: typeof call["id"] === "string" ? call["id"] : "",
        name: typeof fn["name"] === "string" ? fn["name"] : "",
        arguments: typeof fn["arguments"] === "string" ? fn["arguments"] : "",
      });
    });
  }
  return acc;
}

interface ToolCallAcc {
  id: string;
  name: string;
  arguments: string;
}

interface ChatAccumulator {
  id: string | null;
  model: string | null;
  text: string;
  reasoning: string;
  toolCalls: Map<number, ToolCallAcc>;
  finishReason: string | null;
  usage: Record<string, unknown> | null;
  sawDone: boolean;
}

function applyChunk(acc: ChatAccumulator, chunk: Record<string, unknown>): void {
  if (typeof chunk["id"] === "string" && acc.id === null) {
    acc.id = chunk["id"];
  }
  if (typeof chunk["model"] === "string" && acc.model === null) {
    acc.model = chunk["model"];
  }
  if (isRecord(chunk["usage"])) {
    acc.usage = chunk["usage"];
  }
  const choices = chunk["choices"];
  if (!Array.isArray(choices) || choices.length === 0) {
    return;
  }
  const choice = choices[0];
  if (!isRecord(choice)) {
    return;
  }
  if (typeof choice["finish_reason"] === "string") {
    acc.finishReason = choice["finish_reason"];
  }
  const delta = choice["delta"];
  if (!isRecord(delta)) {
    return;
  }
  if (typeof delta["content"] === "string") {
    acc.text += delta["content"];
  }
  const reasoningDelta = delta["reasoning"] ?? delta["reasoning_content"];
  if (typeof reasoningDelta === "string") {
    acc.reasoning += reasoningDelta;
  }
  const toolCalls = delta["tool_calls"];
  if (Array.isArray(toolCalls)) {
    for (const call of toolCalls) {
      if (!isRecord(call)) {
        continue;
      }
      const index = typeof call["index"] === "number" ? call["index"] : 0;
      const entry = acc.toolCalls.get(index) ?? { id: "", name: "", arguments: "" };
      if (typeof call["id"] === "string" && call["id"].length > 0) {
        entry.id = call["id"];
      }
      const fn = call["function"];
      if (isRecord(fn)) {
        if (typeof fn["name"] === "string") {
          entry.name += fn["name"];
        }
        if (typeof fn["arguments"] === "string") {
          entry.arguments += fn["arguments"];
        }
      }
      acc.toolCalls.set(index, entry);
    }
  }
}

// chatUsageToResponses maps chat usage to the Responses shape that
// usageFromResponses reads (camelCase, as the SDK emits it).
function chatUsageToResponses(
  usage: Record<string, unknown> | null
): Record<string, unknown> | null {
  if (usage === null) {
    return null;
  }
  const details = isRecord(usage["completion_tokens_details"])
    ? usage["completion_tokens_details"]
    : undefined;
  return definedValues({
    inputTokens: usage["prompt_tokens"],
    outputTokens: usage["completion_tokens"],
    totalTokens: usage["total_tokens"],
    outputTokensDetails:
      details !== undefined
        ? definedValues({ reasoningTokens: details["reasoning_tokens"] })
        : undefined,
    cost: usage["cost"],
  });
}

// accumulatorToResult builds the ResponsesResult. Output items use the
// SDK's camelCase keys because toResponsesTurn snake_cases them on the
// way out, exactly as it does for a real Responses stream.
export function accumulatorToResult(
  acc: ChatAccumulator,
  generationTimeMs: number
): ResponsesResult {
  const id = acc.id ?? `chatcmpl-${Date.now()}`;
  const output: Record<string, unknown>[] = [];
  if (acc.reasoning.length > 0) {
    output.push({
      type: "reasoning",
      id: `${id}-reasoning`,
      summary: [],
      content: [{ type: "reasoning_text", text: acc.reasoning }],
    });
  }
  if (acc.text.length > 0 || acc.toolCalls.size === 0) {
    output.push({
      type: "message",
      id: `${id}-message`,
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: acc.text, annotations: [] }],
    });
  }
  for (const [, call] of [...acc.toolCalls.entries()].sort((a, b) => a[0] - b[0])) {
    output.push({
      type: "function_call",
      id: `${id}-${call.id}`,
      callId: call.id,
      name: call.name,
      arguments: call.arguments,
      status: "completed",
    });
  }
  const usage = chatUsageToResponses(acc.usage);
  return {
    id,
    model: acc.model ?? "",
    status: acc.finishReason === "length" ? "incomplete" : "completed",
    output,
    usage,
    text: extractMessageText(output),
    generationId: id,
    provider: null,
    generationTimeMs,
  };
}

// readSseStream yields the JSON payload of each `data:` line. Comment
// lines (`: keep-alive`) and blank lines are skipped. `[DONE]` ends the
// stream and is reported through acc.sawDone.
async function readSseStream(
  body: ReadableStream<Uint8Array>,
  onData: (payload: string) => void
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let newline = buffer.indexOf("\n");
    while (newline !== -1) {
      const line = buffer.slice(0, newline).replace(/\r$/u, "");
      buffer = buffer.slice(newline + 1);
      if (line.startsWith("data:")) {
        onData(line.slice(5).trim());
      }
      newline = buffer.indexOf("\n");
    }
  }
  const rest = buffer.trim();
  if (rest.startsWith("data:")) {
    onData(rest.slice(5).trim());
  }
}

// A gateway that fails after committing the 200 reports the failure as an
// OpenAI error object with no HTTP status. OpenRouter's own shape carries
// the status in `code` as a number; SkyPilot Tokens sends the OpenAI shape
// (code null or a string) and names a gateway deadline in the message, so
// the status is taken from `code` when numeric and from the message
// otherwise: 504 for the gateway's own deadline, 502 for a backend cut.
const GATEWAY_DEADLINE_MESSAGE = /went silent past the gateway|deadline/iu;

function streamErrorToResponsesError(
  error: Record<string, unknown>,
  identifiers: ModelErrorIdentifiers
): ResponsesError {
  const code = error["code"];
  const message = typeof error["message"] === "string" ? error["message"] : "stream error";
  const status =
    typeof code === "number" ? code : GATEWAY_DEADLINE_MESSAGE.test(message) ? 504 : 502;
  return new ResponsesError({
    message: appendModelErrorIdentifiers(`Chat stream error: ${message}`, identifiers),
    status,
    retryable: status === 429 || status >= 500,
    ...identifiers,
  });
}

export function makeChatCompletionsLayer(config: ResponsesConfig): Layer<Responses> {
  const traceHeaders = filterTraceHeaders(config.traceHeaders);
  const url = `${chatBaseUrl(config.baseUrl)}/chat/completions`;
  const stream = chatStreamsFromEnv();
  const send = (
    body: ResponsesRequest,
    options: ResponsesSendOptions,
    requestSessionId: string | undefined
  ): Effect<ResponsesResult, ResponsesError> => {
    let identifiers: ModelErrorIdentifiers = {};
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.apiKey}`,
      "HTTP-Referer": BENCH_HARNESS_APP_REFERRER,
      "X-OpenRouter-Title": BENCH_HARNESS_APP_TITLE,
      ...traceHeaders,
      ...options.extraHeaders,
      ...definedValues({ "x-session-id": requestSessionId }),
    };
    return tryPromise({
      try: async (signal) => {
        identifiers = {};
        const startedAt = performance.now();
        const controller = new AbortController();
        const onAbort = () => controller.abort(signal.reason);
        signal.addEventListener("abort", onAbort, { once: true });
        const timer =
          options.timeoutMs !== undefined && options.timeoutMs > 0
            ? setTimeout(
                () => controller.abort(new Error(`Request timed out after ${options.timeoutMs}ms`)),
                options.timeoutMs
              )
            : undefined;
        try {
          const response = await fetch(url, {
            method: "POST",
            headers,
            body: JSON.stringify(responsesRequestToChat(body, options.extraBody, stream)),
            signal: controller.signal,
          });
          identifiers = modelErrorIdentifiersFromFetchHeaders(response.headers);
          options.onResponseIdentifiers?.(identifiers);
          if (!response.ok) {
            const text = await response.text();
            throw new ResponsesError({
              message: appendModelErrorIdentifiers(
                `Chat HTTP ${response.status}: ${text}`,
                identifiers
              ),
              status: response.status,
              retryAfterMs: parseRetryAfter(response.headers.get("retry-after")),
              retryable: response.status === 429 || response.status >= 500,
              providerName: providerNameFromErrorBody(text),
              ...identifiers,
            });
          }
          if (response.body === null) {
            throw new ResponsesError({
              message: appendModelErrorIdentifiers("Empty response body", identifiers),
              retryable: true,
              ...identifiers,
            });
          }
          if (!stream) {
            const acc = nonStreamBodyToAccumulator(await response.text(), identifiers);
            if (acc.id !== null) {
              identifiers = { ...identifiers, generationId: acc.id };
            }
            const result = accumulatorToResult(acc, Math.round(performance.now() - startedAt));
            options.onStreamEvent?.({
              type: "response.completed",
              response: { id: result.id, model: result.model, output: result.output, status: result.status },
            } as unknown as StreamEvents);
            return result;
          }
          const acc: ChatAccumulator = {
            id: null,
            model: null,
            text: "",
            reasoning: "",
            toolCalls: new Map(),
            finishReason: null,
            usage: null,
            sawDone: false,
          };
          let streamError: ResponsesError | undefined;
          await readSseStream(response.body, (payload) => {
            if (payload === "[DONE]") {
              acc.sawDone = true;
              return;
            }
            let parsed: unknown;
            try {
              parsed = JSON.parse(payload);
            } catch {
              return;
            }
            if (!isRecord(parsed)) {
              return;
            }
            if (isRecord(parsed["error"]) && streamError === undefined) {
              streamError = streamErrorToResponsesError(parsed["error"], identifiers);
            }
            applyChunk(acc, parsed);
            if (typeof parsed["id"] === "string") {
              identifiers = { ...identifiers, generationId: parsed["id"] };
            }
          });
          if (streamError !== undefined) {
            throw streamError;
          }
          if (!acc.sawDone && acc.finishReason === null) {
            throw new ResponsesError({
              message: appendModelErrorIdentifiers(
                "Stream ended without [DONE] or a finish_reason",
                identifiers
              ),
              retryable: true,
              ...identifiers,
            });
          }
          const result = accumulatorToResult(acc, Math.round(performance.now() - startedAt));
          // generate() reads response.id off stream events; give it one.
          options.onStreamEvent?.({
            type: "response.completed",
            response: { id: result.id, model: result.model, output: result.output, status: result.status },
          } as unknown as StreamEvents);
          return result;
        } finally {
          if (timer !== undefined) {
            clearTimeout(timer);
          }
          signal.removeEventListener("abort", onAbort);
        }
      },
      catch: (cause) => {
        if (cause instanceof ResponsesError) {
          return cause;
        }
        const message = cause instanceof Error ? cause.message : String(cause);
        const timedOut = /timed out/u.test(message) || (cause instanceof Error && cause.name === "AbortError");
        return new ResponsesError({
          message: appendModelErrorIdentifiers(message, identifiers),
          status: timedOut ? 408 : 500,
          retryable: true,
          ...identifiers,
        });
      },
    }).pipe(
      flatMap((result) => recordGenerationId(result.generationId).pipe(map(() => result)))
    );
  };
  const sendWithSession = (
    body: ResponsesRequest,
    options: ResponsesSendOptions
  ): Effect<ResponsesResult, ResponsesError> =>
    all({ epoch: getCurrentEpoch, retryAttempt: getCurrentRetryAttempt, sampleId: getCurrentSampleId }).pipe(
      flatMap(({ epoch, sampleId }) =>
        send(body, options, buildRequestSessionId(config.sessionId, epoch, sampleId))
      )
    );
  return layerSucceed(Responses, Responses.of({ send: sendWithSession }));
}
