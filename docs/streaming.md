# Streaming

All streaming is driven by `persona.stream(messages)` — a synchronous generator yielding text chunks. Each router wraps this generator in an `async` generator and returns a `StreamingResponse`.

---

## OpenAI — Server-Sent Events

**Endpoint**: `POST /openai/v1/chat/completions` with `"stream": true`

**Media type**: `text/event-stream`

**Format**: Each chunk is prefixed `data: ` followed by a JSON object and `\n\n`. The stream ends with `data: [DONE]\n\n`.

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}\n\n
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n
data: [DONE]\n\n
```

**Source**: `ovos_persona_server/chat.py` — `streaming_chat_response()` inner generator, lines 125–193.

**Usage in final chunk**: Included only if `stream_options.include_usage` is true in the request.

---

## Ollama — Newline-delimited JSON

**Endpoint**: `POST /ollama/api/chat` or `POST /ollama/api/generate` with `"stream": true`

**Media type**: `application/json`

**Format**: Each chunk is a complete JSON object followed by `\n`. No `data:` prefix. The final chunk has `"done": true`.

Chat (`/chat`) chunks:
```
{"model":"...","created_at":"...","message":{"role":"assistant","content":"hello"},"done":false,"total_duration":...}\n
{"model":"...","created_at":"...","message":{"role":"assistant","content":" world"},"done":false}\n
{"model":"...","created_at":"...","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop",...}\n
```

Generate (`/generate`) chunks — note `"response"` key instead of `"message"`:
```
{"model":"...","created_at":"...","response":"hello","done":false}\n
{"model":"...","created_at":"...","response":"","done":true,"done_reason":"stop"}\n
```

**Source**: `ovos_persona_server/ollama.py` — `streaming_ollama_chat_response()` (line 136) and `streaming_ollama_generate_response()` (line 274).

---

## Anthropic — SSE with named events

**Endpoint**: `POST /anthropic/v1/messages` with `"stream": true`

**Media type**: `text/event-stream`

**Format**: Each event has an `event:` line followed by `data:` line and `\n\n`. Event sequence:

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","type":"message","role":"assistant","content":[],"model":"...","stop_reason":null,"usage":{"input_tokens":0,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}

event: message_stop
data: {"type":"message_stop"}
```

**Source**: `ovos_persona_server/anthropic.py` — `_stream()` inner generator, lines 82–115.

---

## Gemini — SSE of GeminiResponse objects

**Endpoint**: `POST /gemini/v1beta/models/{model_id}:streamGenerateContent`

**Media type**: `text/event-stream`

**Format**: Each event is `data: <GeminiResponse JSON>\n\n`. The GeminiResponse has the same structure as the non-streaming response.

```
data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]},"finishReason":"STOP","index":0}]}\n\n
data: {"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]},"finishReason":"STOP","index":0}]}\n\n
```

**Source**: `ovos_persona_server/gemini.py` — `_stream()` inner generator, lines 115–124.

---

## Cohere — Newline-delimited JSON with `event_type`

**Endpoint**: `POST /cohere/v1/chat` with `"stream": true`

**Media type**: `application/json`

**Format**: Newline-delimited JSON objects. Each chunk has an `event_type` field. The final chunk has `event_type: "stream-end"`.

```
{"event_type":"text-generation","text":"hello"}\n
{"event_type":"text-generation","text":" world"}\n
{"event_type":"stream-end","finish_reason":"COMPLETE","response":{"id":"...","text":"hello world","message":{...}}}\n
```

**Source**: `ovos_persona_server/cohere.py` — `_stream()` inner generator, lines 107–128.

---

## HuggingFace TGI — SSE of token events

**Endpoint**: `POST /tgi/generate_stream`

**Media type**: `text/event-stream`

**Format**: Each event is `data:<token JSON>\n\n`. The final event has `"generated_text"` set to the full text and a `"details"` block.

```
data:{"token":{"id":1,"text":"hello","logprob":-1.0,"special":false},"generated_text":null,"details":null}\n\n
data:{"token":{"id":2,"text":" world","logprob":-1.0,"special":false},"generated_text":null,"details":null}\n\n
data:{"token":{"id":0,"text":"","logprob":0.0,"special":true},"generated_text":"hello world","details":{"finish_reason":"eos_token","generated_tokens":2}}\n\n
```

**Source**: `ovos_persona_server/huggingface_tgi.py` — `_stream()` inner generator, lines 93–117.

---

## AWS Bedrock — SSE of outputText events

**Endpoint**: `POST /bedrock/model/{model_id}/invoke-with-response-stream`

**Media type**: `text/event-stream`

**Format**: Each event is `data:<JSON>\n\n`. Chunks have `"outputText"` with the chunk text. The final event has `"completionReason"`.

```
data:{"outputText":"hello","index":0,"totalOutputTextTokenCount":null}\n\n
data:{"outputText":" world","index":0,"totalOutputTextTokenCount":null}\n\n
data:{"outputText":"","index":0,"totalOutputTextTokenCount":2,"completionReason":"FINISH"}\n\n
```

**Source**: `ovos_persona_server/aws_bedrock.py` — `_stream()` inner generator, lines 174–185.
