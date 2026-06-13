# FAQ — `ovos-persona-server`

## General

**What is `ovos-persona-server`?**
A FastAPI server that exposes a single OVOS `Persona` instance as eight concurrent API surfaces: OpenAI, Ollama, Anthropic, Gemini, Cohere, HuggingFace TGI, AWS Bedrock, and A2A (Agent-to-Agent).

## A2A

**How do I expose a persona as an A2A agent?**
```bash
uv pip install 'ovos-persona-server[a2a]'
ovos-persona-server --persona my.json --a2a-base-url http://myhost:8337/a2a
```
The Agent Card is at `GET /a2a/.well-known/agent.json`. JSON-RPC calls go to `POST /a2a/`.

**Is `a2a-sdk` required?**
No — it is an optional dependency. Without it the server starts normally; A2A endpoints are omitted and a warning is logged if `--a2a-base-url` is provided.

**Does the A2A endpoint support streaming?**
Yes. `message/stream` is supported; persona sentence chunks are emitted as `TaskArtifactUpdateEvent` SSE events.

**How do I install it?**
```bash
uv pip install -e .
```
Or from PyPI: `pip install ovos-persona-server`.

**How do I start the server?**
```bash
ovos-persona-server --persona /path/to/persona.json --host 0.0.0.0 --port 8337
```

**What is a persona JSON file?**
```json
{
  "name": "My Assistant",
  "solvers": ["ovos-solver-openai-plugin"],
  "ovos-solver-openai-plugin": {"api_url": "...", "key": "sk-..."}
}
```

**Is authentication required?**
No. All auth headers (`Authorization`, `x-api-key`) are accepted and ignored. Deploy behind a reverse proxy if public access requires auth.

**What Python versions are supported?**
Python >= 3.9 (workspace standard is >= 3.10).

**How do I run tests?**
```bash
uv run pytest test/ -v --cov=ovos_persona_server --cov-report=term-missing
```

---

## API — OpenAI (`/openai/v1`)

**What OpenAI endpoints are available?**
`POST /openai/v1/chat/completions`, `POST /openai/v1/completions`, `GET /openai/v1/models`, `POST /openai/v1/embeddings`.

**Is the `model` field used?**
Accepted but ignored. The persona name loaded at startup is used as the model identifier in all responses.

**What roles are valid in messages?**
`system`, `user`, `assistant`, `tool`, `function`. Invalid roles return HTTP 422.

**Does streaming work for OpenAI?**
Yes. Pass `"stream": true`. Response is `text/event-stream` SSE with `data: {json}\n\n` lines and a terminal `data: [DONE]\n\n`.

**What does `usage` contain?**
Word-split approximations (`len(text.split())`), not real tokenizer counts.

---

## API — Ollama (`/ollama/api`)

**What Ollama endpoints are available?**
`POST /ollama/api/chat`, `POST /ollama/api/generate`, `GET /ollama/api/tags`, `GET /ollama/api/show`, `GET /ollama/api/ps`, `POST /ollama/api/pull`, `POST /ollama/api/push`, `POST /ollama/api/embeddings`.

**What does `/ollama/api/tags` return?**
A single model entry representing the loaded persona, with `family: "ovos-persona"` and `families` set to the loaded solver plugin names. Source: `ovos_persona_server/ollama.py:328`.

**What does `/ollama/api/pull` do?**
It is a stub. Always returns `{"status": "success"}`. No model is actually downloaded.

**Does streaming work for Ollama?**
Yes. Pass `"stream": true`. Response is `application/json` with newline-delimited JSON objects. Final chunk has `"done": true`.

**What is the difference between `/ollama/api/chat` and `/ollama/api/generate`?**
`/chat` uses `messages` (multi-turn). `/generate` uses a single `prompt` string. Streaming chunks from `/generate` use `"response"` key; `/chat` uses `"message"`.

---

## API — Anthropic (`/anthropic/v1`)

**What Anthropic endpoints are available?**
`POST /anthropic/v1/messages`.

**What headers does Anthropic require?**
`x-api-key` is required by the real API but accepted and ignored here. `anthropic-version` is also ignored.

**What happens if `max_tokens` is 0?**
HTTP 422 — `max_tokens` must be >= 1.

**What happens if `messages` is empty?**
HTTP 422 validation error.

**What event types does streaming emit?**
`message_start`, `content_block_start`, `content_block_delta` (one per chunk), `content_block_stop`, `message_delta`, `message_stop`. Source: `ovos_persona_server/anthropic.py:82`.

---

## API — Gemini (`/gemini/v1beta/models`)

**What Gemini endpoints are available?**
`POST /gemini/v1beta/models/{model_id}:generateContent`, `POST /gemini/v1beta/models/{model_id}:streamGenerateContent`.

**How is the `model` role handled?**
Gemini uses `"role": "model"` for assistant turns. This is mapped to `"assistant"` internally before passing to the persona. Source: `ovos_persona_server/gemini.py:38`.

**What does the API key query param do?**
`?key=<api-key>` is accepted and ignored.

**What happens if `contents` is empty?**
HTTP 422 validation error.

---

## API — Cohere (`/cohere/v1`)

**What Cohere endpoints are available?**
`POST /cohere/v1/chat`, `POST /cohere/v1/generate`, `POST /cohere/v1/embed`.

**What does the `/cohere/v1/generate` endpoint return?**
```json
{"id":"...","generations":[{"id":"...","text":"...","finish_reason":"COMPLETE"}]}
```

**What is the temperature range?**
`[0.0, 5.0]`. Values outside this range return HTTP 422.

**What streaming format does Cohere use?**
Newline-delimited JSON with `event_type` field. Chunks have `"event_type": "text-generation"`. Final chunk has `"event_type": "stream-end"`.

---

## API — HuggingFace TGI (`/tgi`)

**What TGI endpoints are available?**
`POST /tgi/generate`, `POST /tgi/generate_stream`, `GET /tgi/info`, `GET /tgi/health`.

**What does `/tgi/info` return?**
A TGI-format model info dict with `model_id` set to the persona name. Other fields are stubs. Source: `ovos_persona_server/huggingface_tgi.py:122`.

**What does `/tgi/health` return?**
HTTP 200 with `{}`.

**What does TGI streaming look like?**
`data:{token:{id,text,logprob,special},...}\n\n` per chunk. Final event has `generated_text` set to full output.

---

## API — AWS Bedrock (`/bedrock/model`)

**What Bedrock endpoints are available?**
`POST /bedrock/model/{model_id}/invoke`, `POST /bedrock/model/{model_id}/invoke-with-response-stream`, `POST /bedrock/model/{model_id}/converse`.

**How does model_id affect the response?**
The response schema varies by `model_id` prefix: `anthropic.claude` → Anthropic format, `meta.llama` → Llama format, `amazon.titan` → Titan format, `cohere.command` → Cohere format, anything else → generic `{"outputText":..., "generated_text":...}`. Source: `ovos_persona_server/aws_bedrock.py:84`.

**Does the Converse API work with all model families?**
Yes. `/converse` uses a typed `BedrockConverseRequest` and always returns a Converse-format response regardless of `model_id`. Source: `ovos_persona_server/aws_bedrock.py:208`.

**Is AWS SigV4 auth verified?**
No. The `Authorization` header is accepted and ignored.

---

## Deprecation

**What are the legacy paths?**
`/v1/...` maps to `/openai/v1/...` and `/api/...` maps to `/ollama/api/...`. Both sets remain fully functional.

**How do I detect deprecated path usage?**
Responses from legacy paths include `Deprecation: true` and `Link: <canonical>; rel="successor-version"` headers. The server also logs a `WARNING`.

**How do I migrate?**
Replace the prefix in your client's base URL: `/v1` → `/openai/v1`, `/api` → `/ollama/api`. No other changes needed.

---

## Embeddings

**Which endpoints support embeddings?**
`POST /openai/v1/embeddings`, `POST /ollama/api/embeddings`, `POST /cohere/v1/embed`.

**What happens if no embeddings solver is configured?**
HTTP 501 with `{"detail": "No embeddings solver configured for this persona."}`.

**How does the server find an embeddings solver?**
It iterates `persona.solvers.loaded_modules` and uses the first solver that has a `get_embeddings` attribute.

---

## Streaming

**Which APIs support streaming?**
OpenAI chat/completions, Ollama chat/generate, Anthropic messages, Gemini streamGenerateContent, Cohere chat/generate, TGI generate_stream, Bedrock invoke-with-response-stream.

**What format does each use?**
See [docs/streaming.md](docs/streaming.md) for per-API format details.

**What happens on a streaming error?**
Each router catches exceptions in the generator and yields an error object before returning. The stream does not silently terminate — an error chunk is emitted.
