
# `ovos-persona-server` — HTTP API Server for OVOS Personas

## What it Does

`ovos-persona-server` exposes a single OVOS `Persona` instance as an HTTP API that is compatible with both the **OpenAI Chat Completions API** and the **Ollama API**. Any client that can talk to OpenAI or Ollama (e.g. Open WebUI, LangChain, llm CLI) can use it without modification.

The server loads one persona from a JSON file at startup and serves all requests through it.

---

## Running the Server

### Install

```bash
pip install ovos-persona-server
```

### Start

```bash
ovos-persona-server --persona /path/to/persona.json --host 0.0.0.0 --port 8337
```

CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--persona` | required | Path to the persona `.json` file |
| `--host` | `0.0.0.0` | Host address to bind |
| `--port` | `8337` | TCP port to listen on |

The server is built with **FastAPI** and served by **Uvicorn**. CORS is enabled for all origins, methods, and headers.

### Persona JSON Format

```json
{
  "name": "My Persona",
  "solvers": ["ovos-solver-openai-plugin"],
  "ovos-solver-openai-plugin": {
    "api_url": "https://llama.smartgic.io/v1",
    "key": "sk-xxxx",
    "persona": "helpful and friendly."
  }
}
```

See `ovos-persona` documentation for full persona JSON schema.

---

## Authentication

There is no authentication. The server accepts requests from any origin. Deploy behind a reverse proxy with authentication if the endpoint is publicly accessible.

---

## HTTP API Endpoints

All endpoints are provided by two FastAPI routers:

- `chat_router` — prefix `/v1`, OpenAI-compatible
- `ollama_router` — prefix `/api`, Ollama-compatible

### OpenAI-Compatible Endpoints (`/v1`)

#### `POST /v1/chat/completions`

OpenAI Chat Completions API. Supports both streaming and non-streaming responses.

Request body (`CreateChatCompletionRequest`):

| Field | Type | Notes |
|-------|------|-------|
| `messages` | `List[ChatCompletionRequestMessage]` | Conversation history; only `messages` and `stream` are acted on |
| `stream` | `bool` | Default `false`. If `true`, returns `text/event-stream` SSE chunks |
| `model` | `str` | Accepted but ignored; persona name is used as the model identifier in responses |
| `temperature`, `top_p`, `seed`, etc. | various | Accepted, echoed in response, but not passed to underlying solvers |

Non-streaming response (`CreateChatCompletionResponse`):

```json
{
  "id": "chatcmpl-<28 chars>",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "<persona name>",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "<answer>"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": N, "completion_tokens": M, "total_tokens": N+M}
}
```

Streaming response: Server-Sent Events, each line prefixed `data: `, terminated with `data: [DONE]`. Each chunk uses `object: "chat.completion.chunk"`.

Error: `HTTP 500` with `detail` on persona failure.

#### `POST /v1/completions`

Legacy OpenAI Completions API (text completion, not chat).

Request body (`CreateCompletionRequest`):

| Field | Type | Notes |
|-------|------|-------|
| `prompt` | `str \| List[str]` | Converted to a `user` message for the persona. Token ID formats (`List[int]`) return HTTP 500 |
| `stream` | `bool` | Default `false` |
| `model` | `str` | Echoed in response |

Non-streaming response (`CreateCompletionResponse`):

```json
{
  "id": "cmpl-<28 chars>",
  "object": "text_completion",
  "created": 1234567890,
  "model": "<requested model>",
  "choices": [{"text": "<answer>", "index": 0, "logprobs": null, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": N, "completion_tokens": M, "total_tokens": N+M}
}
```

---

### Ollama-Compatible Endpoints (`/api`)

#### `POST /api/chat`

Ollama Chat API.

Request body (`OllamaChatRequest`):

| Field | Type | Notes |
|-------|------|-------|
| `messages` | `List[OllamaChatMessage]` | Required. Each message has `role`, `content`, optional `images`, `tool_calls`, `thinking` |
| `stream` | `bool` | Default `false`. Streaming returns newline-delimited JSON objects |
| `model` | `str` | Accepted but ignored |
| `tools`, `format`, `options`, `keep_alive`, `think` | various | Accepted but not currently used |

Non-streaming response (`OllamaChatResponse`):

```json
{
  "model": "<persona name>",
  "created_at": "2026-03-08T12:00:00.000000Z",
  "message": {"role": "assistant", "content": "<answer>"},
  "done": true,
  "total_duration": 1234567890,
  "done_reason": "stop"
}
```

Streaming response: each chunk is a JSON object followed by `\n`. Chunks have `"done": false`; the final chunk has `"done": true` with `done_reason`.

Lang and units come from the default OVOS session (`SessionManager().get()`).

#### `POST /api/generate`

Ollama Generate API (single-turn prompt, not multi-turn chat).

Request body (`OllamaGenerateRequest`):

| Field | Type | Notes |
|-------|------|-------|
| `prompt` | `str` | Required |
| `system` | `str` | Optional system prompt; prepended as a `system` role message |
| `stream` | `bool` | Default `false` |
| `suffix`, `images`, `format`, `options`, `template`, `raw`, `keep_alive`, `context`, `think` | various | Accepted but not currently used |

Response uses the same `OllamaChatResponse` schema as `/api/chat`. Streaming chunks use `"response"` key (not `"message"`) per Ollama spec.

#### `GET /api/tags`

List available models (Ollama model listing).

Response (`OllamaTagsResponse`):

```json
{
  "models": [{
    "name": "<persona name>",
    "model": "<persona name>",
    "digest": "sha256:placeholder_digest",
    "size": 0,
    "modified_at": "<ISO timestamp>",
    "details": {
      "format": "json",
      "family": "ovos-persona",
      "families": ["<solver plugin name>", ...],
      "parent_model": "<model names from solver configs>",
      "parameter_size": "",
      "quantization_level": ""
    }
  }]
}
```

Returns one entry for the loaded persona. `families` reflects the loaded solver plugin names. `parent_model` is a `|`-joined list of `model` keys from solver configs if present.

---

## Request/Response Schemas

All schemas are Pydantic v2 models defined in:

- `ovos_persona_server.schemas.openai_chat` — `CreateChatCompletionRequest`, `CreateChatCompletionResponse`, `CreateChatCompletionStreamResponse`, `CreateCompletionRequest`, `CreateCompletionResponse`, `CompletionUsage`, `FinishReason`, `Role`, etc.
- `ovos_persona_server.schemas.ollama` — `OllamaChatRequest`, `OllamaChatResponse`, `OllamaGenerateRequest`, `OllamaTagsResponse`, `OllamaModel`, `OllamaModelDetails`, `OllamaChatMessage`, `OllamaEmbedRequest`, `OllamaEmbedResponse`

Token counts in `usage` are word-split approximations (`len(text.split())`), not real tokenizer counts.

---

## Internal Structure

| Module | Role |
|--------|------|
| `ovos_persona_server/__init__.py` — `create_persona_app(persona_path)` | Loads persona JSON, sets `default_persona` global, creates FastAPI app, registers routers |
| `ovos_persona_server/persona.py` — `get_default_persona()` | FastAPI dependency; returns the loaded `Persona` instance or raises HTTP 500 |
| `ovos_persona_server/chat.py` — `chat_router` | OpenAI `/v1` endpoints |
| `ovos_persona_server/ollama.py` — `ollama_router` | Ollama `/api` endpoints |
| `ovos_persona_server/__main__.py` — `main()` | CLI entry point using argparse + uvicorn |

---

## Cross-References

- `ovos-persona` — `Persona` class, solver plugin chain, memory
- `ovos-bus-client` — `SessionManager` (used by Ollama endpoints to read default lang/units)
- `ovos-plugin-manager` — solver plugin discovery (indirect, via `ovos-persona`)
