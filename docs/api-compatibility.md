# API Compatibility

All seven API surfaces share a single backing `Persona` instance loaded at startup. The `model` field in requests is accepted but ignored — the persona name is used as the model identifier in all responses. Auth headers are accepted and ignored.

## 1. OpenAI — `/openai/v1`

**Source**: `ovos_persona_server/chat.py` — `chat_router`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/openai/v1/chat/completions` | Chat completions (streaming supported) |
| POST | `/openai/v1/completions` | Legacy text completions (streaming supported) |
| GET | `/openai/v1/models` | Model listing |
| POST | `/openai/v1/embeddings` | Embeddings (requires solver — see [embeddings.md](embeddings.md)) |

**Auth**: `Authorization: Bearer <token>` — accepted, ignored.

**Request schema** (`POST /openai/v1/chat/completions`):
```json
{
  "model": "any-string",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}
```
Valid roles: `system`, `user`, `assistant`, `tool`, `function`. Invalid roles return 422.

**Client system prompt** (`system_prompt_strategy`): a client-supplied `system` message is
handled per the persona's `system_prompt_strategy`, set with the
`PERSONA_SYSTEM_PROMPT_STRATEGY` environment variable or the `system_prompt_strategy` key in
the persona JSON:

| Strategy | Effect |
| :--- | :--- |
| `ignore` (default) | Drop the client's system message; use only the persona's own `system_prompt`. Preserves existing behaviour. |
| `replace` | Use the client's system message(s) instead of the persona's; fall back to the persona's when the client sends none. |
| `append` | Persona's `system_prompt` first, then the client's system message(s) appended after it, joined by a blank line. |

Multiple client system messages are concatenated in order. The strategy is applied once to the
incoming request, so it holds for every path.

**Response schema**:
```json
{
  "id": "chatcmpl-<28 chars>",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "<persona name>",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": N, "completion_tokens": M, "total_tokens": N+M}
}
```

**Streaming**: See [streaming.md](streaming.md).

**curl example**:
```bash
curl -s http://localhost:8337/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-persona","messages":[{"role":"user","content":"Hello"}]}'
```

---

## 2. Ollama — `/ollama/api`

**Source**: `ovos_persona_server/ollama.py` — `ollama_router`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/ollama/api/chat` | Multi-turn chat (streaming supported) |
| POST | `/ollama/api/generate` | Single-turn generation (streaming supported) |
| GET | `/ollama/api/tags` | List models |
| GET | `/ollama/api/show` | Model card |
| GET | `/ollama/api/ps` | Running models |
| POST | `/ollama/api/pull` | Pull model (stub — always returns `{"status":"success"}`) |
| POST | `/ollama/api/push` | Push model (stub — always returns `{"status":"success"}`) |
| POST | `/ollama/api/embeddings` | Embeddings (requires solver — see [embeddings.md](embeddings.md)) |

**Auth**: None required.

**Request schema** (`POST /ollama/api/chat`):
```json
{
  "model": "any-string",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}
```

**Response schema** (non-streaming):
```json
{
  "model": "<persona name>",
  "created_at": "2026-03-18T12:00:00.000000Z",
  "message": {"role": "assistant", "content": "..."},
  "done": true,
  "total_duration": 123456789,
  "done_reason": "stop"
}
```

**curl example**:
```bash
curl -s http://localhost:8337/ollama/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"my-persona","messages":[{"role":"user","content":"Hello"}]}'
```

---

## 3. Anthropic — `/anthropic/v1`

**Source**: `ovos_persona_server/anthropic.py` — `anthropic_router`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/anthropic/v1/messages` | Create message (streaming supported) |

**Auth**: `x-api-key: <key>` — accepted, ignored.

**Request schema**:
```json
{
  "model": "claude-3-opus-20240229",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 1024
}
```
`messages` must be non-empty and `max_tokens` must be >= 1. Content may be a string or list of `{"type":"text","text":"..."}` blocks.

**Response schema**:
```json
{
  "id": "msg_<24 chars>",
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "..."}],
  "model": "<requested model>",
  "stop_reason": "end_turn",
  "usage": {"input_tokens": N, "output_tokens": M}
}
```

**curl example**:
```bash
curl -s http://localhost:8337/anthropic/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: fake-key" \
  -d '{"model":"claude-3-opus-20240229","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

---

## 4. Gemini — `/gemini/v1beta/models`

**Source**: `ovos_persona_server/gemini.py` — `gemini_router`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/gemini/v1beta/models/{model_id}:generateContent` | Generate content |
| POST | `/gemini/v1beta/models/{model_id}:streamGenerateContent` | Stream content |

**Auth**: `?key=<api-key>` query param — accepted, ignored.

**Request schema**:
```json
{
  "contents": [
    {"role": "user", "parts": [{"text": "Hello"}]}
  ]
}
```
`contents` must be non-empty. `role` may be `user` or `model` (mapped to `assistant` internally).

**Response schema**:
```json
{
  "candidates": [
    {
      "content": {"role": "model", "parts": [{"text": "..."}]},
      "finishReason": "STOP",
      "index": 0
    }
  ]
}
```

**curl example**:
```bash
curl -s "http://localhost:8337/gemini/v1beta/models/gemini-pro:generateContent?key=fake" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}'
```

---

## 5. Cohere — `/cohere/v1`

**Source**: `ovos_persona_server/cohere.py` — `cohere_router`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/cohere/v1/chat` | Chat (streaming supported) |
| POST | `/cohere/v1/generate` | Generate text (streaming supported) |
| POST | `/cohere/v1/embed` | Embeddings (requires solver — see [embeddings.md](embeddings.md)) |

**Auth**: `Authorization: Bearer <token>` — accepted, ignored.

**Request schema** (`POST /cohere/v1/chat`):
```json
{
  "message": "Hello",
  "stream": false
}
```
`message` must be non-empty. `temperature` must be in `[0.0, 5.0]`.

**Response schema**:
```json
{
  "id": "<24 chars>",
  "finish_reason": "COMPLETE",
  "text": "...",
  "message": {"role": "assistant", "content": [{"type": "text", "text": "..."}]},
  "usage": {"billed_units": {"input_tokens": N, "output_tokens": M}}
}
```

**curl example**:
```bash
curl -s http://localhost:8337/cohere/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello"}'
```

---

## 6. HuggingFace TGI — `/tgi`

**Source**: `ovos_persona_server/huggingface_tgi.py` — `tgi_router`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/tgi/generate` | Generate text |
| POST | `/tgi/generate_stream` | Stream text (SSE) |
| GET | `/tgi/info` | Model info |
| GET | `/tgi/health` | Health check |

**Auth**: None required.

**Request schema** (`POST /tgi/generate`):
```json
{
  "inputs": "Once upon a time",
  "parameters": {
    "max_new_tokens": 256,
    "temperature": 0.7
  }
}
```
`inputs` must be non-empty.

**Response schema**:
```json
{
  "generated_text": "...",
  "details": {"finish_reason": "eos_token", "generated_tokens": N}
}
```

**curl example**:
```bash
curl -s http://localhost:8337/tgi/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs":"Once upon a time"}'
```

---

## 7. AWS Bedrock — `/bedrock/model`

**Source**: `ovos_persona_server/aws_bedrock.py` — `bedrock_router`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/bedrock/model/{model_id}/invoke` | Invoke model |
| POST | `/bedrock/model/{model_id}/invoke-with-response-stream` | Invoke with streaming |
| POST | `/bedrock/model/{model_id}/converse` | Converse API |

**Auth**: `Authorization: AWS4-HMAC-SHA256 ...` — accepted, ignored.

**Request schema** varies by model family. See [bedrock-models.md](bedrock-models.md).

**curl example** (Anthropic format):
```bash
curl -s http://localhost:8337/bedrock/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```
