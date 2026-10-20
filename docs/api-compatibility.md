# API Compatibility

## What you can do

You have code written for the OpenAI SDK, or for the Anthropic, Ollama, Gemini, Cohere, Bedrock or Hugging Face TGI clients. Change the base URL in that code to this server. The persona answers, through the chat engine and the knowledge that its config names. The SDK, the request and the response stay the same.

The OpenAI surface also gives function calling, embeddings, files and vector stores, so code that uses those calls works too.

## User stories

Each recipe below was run with the official SDK against `ovos-persona-server --persona llm.json`. The persona uses `ovos-chat-openai-plugin` with `qwen2.5:7b` on an Ollama host. The output lines are the real output.

```json
{
  "name": "llm",
  "solvers": ["ovos-chat-openai-plugin"],
  "ovos-chat-openai-plugin": {
    "api_url": "https://<your-ollama-host>/v1",
    "key": "unused",
    "model": "qwen2.5:7b",
    "system_prompt": "You are a helpful assistant who gives short, factual answers."
  }
}
```

### I have code for OpenAI chat completions

Set `base_url` to the `/openai/v1` prefix. `model` is the persona name.

```python
from openai import OpenAI

client = OpenAI(api_key="unused", base_url="http://localhost:8337/openai/v1")
print([m.id for m in client.models.list().data])
reply = client.chat.completions.create(
    model="llm", messages=[{"role": "user", "content": "What is the capital of Portugal? One word."}])
print(reply.choices[0].message.content)
```

```text
['llm']
Lisbon.
```

`stream=True` also works. See [streaming.md](streaming.md).

### My code calls my own functions through `tools`

The model asks for the call, your code runs the function, and the model writes the answer from the result. The persona's chat engine must support tools. `ovos-chat-openai-plugin` supports them.

```python
import json

def get_weather(city):
    return {"city": city, "temp_c": 21, "sky": "sunny"}

tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

messages = [{"role": "user", "content": "What's the weather in Lisbon right now?"}]
first = client.chat.completions.create(model="llm", messages=messages, tools=tools)
call = first.choices[0].message.tool_calls[0]
print(first.choices[0].finish_reason, call.function.name, call.function.arguments)

messages.append(first.choices[0].message.model_dump(exclude_none=True))
messages.append({"role": "tool", "tool_call_id": call.id,
                 "content": json.dumps(get_weather(**json.loads(call.function.arguments)))})
print(client.chat.completions.create(model="llm", messages=messages, tools=tools).choices[0].message.content)
```

```text
tool_calls get_weather {"city": "Lisbon"}
The current weather in Lisbon is sunny with a temperature of 21 degrees Celsius.
```

With `stream=True`, the tool call arrives as one `tool_calls` delta. `tool_choice="none"` and `tool_choice={"type": "function", "function": {"name": ...}}` work. `tool_choice="required"` gets a 422 error, because the OVOS chat engine contract cannot force the model to call a tool.

### I need embeddings

```python
result = client.embeddings.create(model="x", input=["cats purr", "the moon"])
print(len(result.data), len(result.data[0].embedding))
```

```text
2 384
```

`encoding_format="base64"` also works. The size of the vector comes from the embeddings plugin. [embeddings.md](embeddings.md) has the details.

### I search my own documents with files and vector stores

Upload files, put them in a vector store, and search it. The server chunks and embeds the files. [rag.md](rag.md) has the full story and the configuration.

```python
cats = client.files.create(file=("cats.txt", b"the cat sat on the mat. cats are fluffy animals that purr."),
                           purpose="assistants")
moon = client.files.create(file=("moon.txt", b"the moon orbits the earth and affects ocean tides."),
                           purpose="assistants")
store = client.vector_stores.create(name="kb")
for f in (cats, moon):
    client.vector_stores.files.create(vector_store_id=store.id, file_id=f.id)
for hit in client.vector_stores.search(vector_store_id=store.id, query="fluffy animal sitting on a mat").data:
    print(hit.filename, round(hit.score, 4))
```

```text
cats.txt 0.7249
moon.txt 0.0298
```

`score` is a similarity: higher is closer.

### I have code for another vendor SDK

These SDKs asked the same persona "What is the capital of Portugal? One word.":

| SDK | Setting | Call | Output |
| :--- | :--- | :--- | :--- |
| `anthropic` 1.15.0 | `Anthropic(base_url="http://localhost:8337/anthropic")` | `messages.create(model="llm", max_tokens=50, messages=[...])` | `Lisbon.` |
| `ollama` 0.6.3 | `ollama.Client(host="http://localhost:8337/ollama")` | `chat(model="llm", messages=[...])` | `Lisbon.` |
| `google-genai` 2.31.0 | `genai.Client(http_options=types.HttpOptions(base_url="http://localhost:8337/gemini"))` | `models.generate_content(model="llm", contents=...)` | `Lisbon.` |
| `cohere` 7.2.0 | `cohere.Client(base_url="http://localhost:8337/cohere")` | `chat(model="llm", message=...)` | `Lisbon.` |
| `boto3` 1.43.115 | `boto3.client("bedrock-runtime", endpoint_url="http://localhost:8337/bedrock")` | `converse(modelId="llm", messages=[...])` | `Lisbon.` |
| `huggingface_hub` 2.2.0 | `InferenceClient(model="http://localhost:8337/tgi")` | `text_generation(...)` | `Lisbon.` |

## Endpoint status

"Live" means that the SDK in the table was run against a server with a real LLM persona, as in the stories above. "Refused" means that the server answers with an error that says why.

| Surface | Call | Status |
| :--- | :--- | :--- |
| OpenAI | `models.list`, `chat.completions.create`, streaming | Live |
| OpenAI | `tools` with `tool_calls`, plain and streaming | Live |
| OpenAI | `tool_choice="required"` | Refused, 422 |
| OpenAI | `embeddings.create`, float and base64 | Live |
| OpenAI | `files` create, list, retrieve, content, delete | Live |
| OpenAI | `vector_stores` create, files, list, retrieve, search | Live |
| Anthropic | `messages.create` | Live |
| Ollama | `chat` | Live |
| Gemini | `models.generate_content` | Live |
| Cohere | `chat` | Live |
| AWS Bedrock | `converse` | Live |
| Hugging Face TGI | `text_generation` | Live |
| Hugging Face TGI | `chat_completion` (`/v1/chat/completions`) | Not served, 404 |

## Reference

All seven API surfaces share the personas loaded at startup. The `model` field selects one of them: a persona's `name` is its model id, and the persona name is used as the model identifier in responses. With a single persona loaded, the field is accepted and ignored. With several personas, an unknown name returns HTTP 404 and an omitted field selects the default persona. [multi-persona.md](multi-persona.md) gives the details per surface. Auth headers are accepted and ignored.

### 1. OpenAI: `/openai/v1`

**Source**: `ovos_persona_server/chat.py`, `chat_router`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/openai/v1/chat/completions` | Chat completions (streaming supported) |
| POST | `/openai/v1/completions` | Legacy text completions (streaming supported) |
| GET | `/openai/v1/models` | Model listing |
| POST | `/openai/v1/embeddings` | Embeddings (requires a solver, see [embeddings.md](embeddings.md)) |

**Auth**: `Authorization: Bearer <token>`, accepted and ignored.

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
| `ignore` (default) | Drop the client's system message. Use only the persona's own `system_prompt`. |
| `replace` | Use the client's system message(s) instead of the persona's. Fall back to the persona's when the client sends none. |
| `append` | Persona's `system_prompt` first, then the client's system message(s) appended after it, joined by a blank line. |

Multiple client system messages are concatenated in order. The strategy is applied once to the
incoming request, so it holds for every path.

The strategy runs **before** the chat engine sees the messages, so it wins over any
engine-level setting. `ovos-chat-openai-plugin` has its own `allow_system_prompts`
(default `false`, which already strips client system messages). A persona that set it
to `true` to merge the client's prompt keeps receiving nothing under the default
`ignore`. Such a persona wants `append`, which reproduces that merge at the server
level: persona prompt first, client's after it.

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

### 2. Ollama: `/ollama/api`

**Source**: `ovos_persona_server/ollama.py`, `ollama_router`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/ollama/api/chat` | Multi-turn chat (streaming supported) |
| POST | `/ollama/api/generate` | Single-turn generation (streaming supported) |
| GET | `/ollama/api/tags` | List models |
| GET | `/ollama/api/show` | Model card |
| GET | `/ollama/api/ps` | Running models |
| POST | `/ollama/api/pull` | Pull model (stub, always returns `{"status":"success"}`) |
| POST | `/ollama/api/push` | Push model (stub, always returns `{"status":"success"}`) |
| POST | `/ollama/api/embeddings` | Embeddings (requires a solver, see [embeddings.md](embeddings.md)) |

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

### 3. Anthropic: `/anthropic/v1`

**Source**: `ovos_persona_server/anthropic.py`, `anthropic_router`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/anthropic/v1/messages` | Create message (streaming supported) |

**Auth**: `x-api-key: <key>`, accepted and ignored.

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

### 4. Gemini: `/gemini/v1beta/models`

**Source**: `ovos_persona_server/gemini.py`, `gemini_router`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/gemini/v1beta/models/{model_id}:generateContent` | Generate content |
| POST | `/gemini/v1beta/models/{model_id}:streamGenerateContent` | Stream content |

**Auth**: `?key=<api-key>` query param, accepted and ignored.

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

### 5. Cohere: `/cohere/v1`

**Source**: `ovos_persona_server/cohere.py`, `cohere_router`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/cohere/v1/chat` | Chat (streaming supported) |
| POST | `/cohere/v1/generate` | Generate text (streaming supported) |
| POST | `/cohere/v1/embed` | Embeddings (requires a solver, see [embeddings.md](embeddings.md)) |

**Auth**: `Authorization: Bearer <token>`, accepted and ignored.

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

### 6. HuggingFace TGI: `/tgi`

**Source**: `ovos_persona_server/huggingface_tgi.py`, `tgi_router`

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

### 7. AWS Bedrock: `/bedrock/model`

**Source**: `ovos_persona_server/aws_bedrock.py`, `bedrock_router`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/bedrock/model/{model_id}/invoke` | Invoke model |
| POST | `/bedrock/model/{model_id}/invoke-with-response-stream` | Invoke with streaming |
| POST | `/bedrock/model/{model_id}/converse` | Converse API |

**Auth**: `Authorization: AWS4-HMAC-SHA256 ...`, accepted and ignored.

**Request schema** varies by model family. See [bedrock-models.md](bedrock-models.md).

**curl example** (Anthropic format):
```bash
curl -s http://localhost:8337/bedrock/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

## Deployment

The server listens on port 8337 by default. It accepts any API key and ignores it, so put it behind a reverse proxy that checks the key. Every surface shares the personas that the server loads. [multi-persona.md](multi-persona.md) shows how `model` selects one.

## Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `tools` gets 501 | The persona's chat engine does not support tools | Use a tool-capable engine, such as `ovos-chat-openai-plugin` |
| `tool_choice="required"` gets 422 | The chat engine contract cannot force a tool call | Send `tool_choice="auto"`, or name the function |
| `InferenceClient.chat_completion` gets 404 | The TGI surface serves `text_generation` only | Use `text_generation`, or the OpenAI surface for chat |
| A client system message has no effect | `system_prompt_strategy` is `ignore` by default | Set `PERSONA_SYSTEM_PROMPT_STRATEGY` to `append` or `replace` |
