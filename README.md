# ovos-persona-server

A single HTTP server that exposes one or more OVOS `Persona`s as **eight concurrent API surfaces** — so any LLM client (OpenAI SDK, LangChain, Ollama tools, Anthropic SDK, Google Gemini SDK, Cohere SDK, HuggingFace TGI client, AWS Bedrock client, or any A2A agent) can talk to your OVOS persona without changes.

---

## Table of Contents

- [What is a Persona?](#what-is-a-persona)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Serving Several Personas](#serving-several-personas)
- [API Surfaces](#api-surfaces)
- [A2A Endpoint](#a2a-endpoint)
- [Persona Config Examples](#persona-config-examples)
- [Streaming](#streaming)
- [RAG: Files & Vector Stores](#rag-files--vector-stores)
- [Embeddings](#embeddings)
- [Authentication](#authentication)
- [Troubleshooting](#troubleshooting)

---

## What is a Persona?

An OVOS Persona is a JSON file that chains together one or more **solver plugins**. Solvers are tried in order until one returns an answer. You can mix LLMs, knowledge bases, and fallback bots in a single persona — no GPU required for non-LLM setups.

```json
{
  "name": "OldSchoolBot",
  "solvers": [
    "ovos-solver-wikipedia-plugin",
    "ovos-solver-ddg-plugin",
    "ovos-solver-plugin-wolfram-alpha",
    "ovos-solver-wordnet-plugin",
    "ovos-solver-rivescript-plugin",
    "ovos-solver-failure-plugin"
  ],
  "ovos-solver-plugin-wolfram-alpha": { "appid": "YOUR_API_KEY" }
}
```

Find solver plugins at [github.com/OpenVoiceOS](https://github.com/OpenVoiceOS?q=solver).

---

## Installation

```bash
# Base server (no A2A)
pip install ovos-persona-server

# With A2A server support
pip install 'ovos-persona-server[a2a]'
```

With `uv` (recommended in OVOS workspaces):

```bash
uv pip install 'ovos-persona-server[a2a]'
```

---

## Quick Start

```bash
# Start serving a persona on port 8337
ovos-persona-server --persona /path/to/my-persona.json

# Also expose it as an A2A agent
ovos-persona-server \
  --persona /path/to/my-persona.json \
  --a2a-base-url http://localhost:8337/a2a
```

The server binds to `0.0.0.0:8337` by default. Visit `http://localhost:8337/docs` for the interactive API reference (Swagger UI).

---

## Serving Several Personas

One process can host several personas. A persona's `name` is its model id: clients select one with the `model` field of whichever API they speak.

```bash
ovos-persona-server \
  --persona /path/to/assistant.json \
  --persona /path/to/rivescript-bot.json

# or load a whole directory
ovos-persona-server --personas-dir /etc/ovos/personas --default-persona assistant
```

| Flag | Purpose |
|------|---------|
| `--persona` | Path to a persona `.json`. Repeat it to load more than one. |
| `--personas-dir` | Load every `*.json` in the directory, sorted by file name. |
| `--default-persona` | Persona used when a request names no model. Defaults to the first persona loaded. |

`GET /openai/v1/models` (and `GET /ollama/api/tags`) list every loaded persona. An unknown `model` returns HTTP 404 listing the available names. With a single persona the `model` field stays advisory, so existing deployments are unaffected.

`model` selects a **persona** — a solver chain, a system prompt, and memory settings — not the LLM behind it. To change the LLM, edit the persona JSON.

Per-surface behaviour, the default-persona rule, and how state is kept separate: [docs/multi-persona.md](docs/multi-persona.md).

---

## API Surfaces

Every API is served on a vendor-prefixed path so multiple clients can coexist without conflict.

| API | Prefix | Key endpoints |
|-----|--------|---------------|
| OpenAI | `/openai/v1` | `POST /chat/completions`, `POST /completions`, `GET /models`, `POST /embeddings` |
| OpenAI RAG | `/openai/v1` | `…/files`, `…/vector_stores`, `…/vector_stores/{id}/search` — see [RAG](#rag-files--vector-stores) |
| Ollama | `/ollama/api` | `POST /chat`, `POST /generate`, `GET /tags`, `POST /embed`, `POST /embeddings` |
| Anthropic | `/anthropic/v1` | `POST /messages` |
| Google Gemini | `/gemini/v1beta/models` | `POST /{model}:generateContent`, `:streamGenerateContent`, `:embedContent`, `:batchEmbedContents` |
| Cohere | `/cohere/v1` | `POST /chat`, `POST /generate`, `POST /embed` |
| HuggingFace TGI | `/tgi` | `POST /generate`, `POST /generate_stream`, `POST /embed` |
| AWS Bedrock | `/bedrock/model` | `POST /{model}/invoke` (chat + Titan/Cohere embed), `POST /{model}/invoke-with-response-stream` |
| A2A | `/a2a` | `GET /.well-known/agent.json`, `POST /` |

### Deprecated legacy paths

For backwards compatibility, `/v1/...` maps to `/openai/v1/...` and `/api/...` maps to `/ollama/api/...`. These paths send `Deprecation` and `Link` response headers and will be removed in a future major version. Migrate to the prefixed paths.

### Quick test with curl

```bash
# OpenAI-compatible chat
curl -s http://localhost:8337/openai/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"","messages":[{"role":"user","content":"hello"}]}' \
  | python3 -m json.tool

# Ollama-compatible chat
curl -s http://localhost:8337/ollama/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"","messages":[{"role":"user","content":"hello"}]}'
```

---

## A2A Endpoint

`ovos-persona-server` can expose your persona as a standard [A2A](https://google.github.io/A2A/) agent server, enabling any A2A client to interact with it — including **ovos-a2a-agent** running on another OVOS instance.

### Enable A2A

```bash
ovos-persona-server \
  --persona my-persona.json \
  --a2a-base-url http://myhost:8337/a2a
```

The `--a2a-base-url` flag:
- Activates the A2A endpoint at `/a2a`.
- Sets the `url` field in the Agent Card returned at `GET /a2a/.well-known/agent.json`.
- Must be the **publicly reachable** URL of the `/a2a` mount — this is what A2A clients use to discover the server.

### Verify

```bash
# Fetch the Agent Card
curl http://localhost:8337/a2a/.well-known/agent.json | python3 -m json.tool

# Send a message
curl -X POST http://localhost:8337/a2a/ \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "hello"}]
      }
    }
  }'
```

### Connecting ovos-a2a-agent to this server

On another OVOS instance:

```json
{
  "name": "remote-persona",
  "chat_module": "ovos-a2a-agent",
  "ovos-a2a-agent": {
    "url": "http://myhost:8337/a2a"
  }
}
```

### A2A streaming

The A2A endpoint supports `message/stream`. Persona sentence chunks are emitted as `TaskArtifactUpdateEvent` SSE events. Enable streaming on the client side (e.g. `"streaming": true` in `ovos-a2a-agent` config).

### A2A without `a2a-sdk`

If `a2a-sdk` is not installed and `--a2a-base-url` is provided, the server starts normally and logs a warning. All other API surfaces continue to work.

---

## Persona Config Examples

### LLM persona (OpenAI-compatible backend)

```json
{
  "name": "gpt-persona",
  "chat_module": "ovos-openai-plugin",
  "ovos-openai-plugin": {
    "api_key": "sk-...",
    "model": "gpt-4o-mini"
  }
}
```

### Knowledge-base + LLM fallback

```json
{
  "name": "smart-assistant",
  "solvers": [
    "ovos-solver-wikipedia-plugin",
    "ovos-solver-ddg-plugin",
    "ovos-solver-wordnet-plugin",
    "ovos-openai-plugin",
    "ovos-solver-failure-plugin"
  ],
  "ovos-openai-plugin": {
    "api_key": "sk-...",
    "model": "gpt-4o-mini"
  }
}
```

### Rivescript chatbot (no GPU, no API key)

```json
{
  "name": "rivescript-bot",
  "solvers": [
    "ovos-solver-rivescript-plugin",
    "ovos-solver-failure-plugin"
  ]
}
```

---

## Streaming

All seven non-A2A APIs support SSE streaming where the upstream spec defines it. Pass `"stream": true` (OpenAI / Cohere / TGI) or the equivalent for each API. See [docs/streaming.md](docs/streaming.md) for per-API details.

---

## OPM Tool Plugins — MCP and UTCP exposure

Installed `ToolBox` plugins (OPM entry-point group `opm.agents.toolbox`) are
automatically surfaced over two protocols when the server starts.

### Installing the MCP extra

```bash
pip install ovos-persona-server[mcp]
```

Without the `[mcp]` extra only the UTCP endpoints are active.

### UTCP — Universal Tool Calling Protocol

Two endpoints are added at `/tools`:

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/tools/manual` | Returns a UTCP manual JSON listing all tools |
| `POST` | `/tools/{name}` | Invoke a tool by name with a JSON body |

**Fetch the manual:**

```bash
curl http://localhost:8337/tools/manual
```

Response shape:

```json
{
  "utcp_version": "1.0",
  "tools": [
    {
      "name": "my_tool",
      "description": "Does something useful.",
      "tool_provider": {
        "type": "http",
        "method": "POST",
        "url": "http://localhost:8337/tools/my_tool",
        "content_type": "application/json"
      },
      "inputs": [
        {"name": "query", "type": "string", "required": true, "description": "Search query"}
      ],
      "output_schema": { ... }
    }
  ]
}
```

**Invoke a tool:**

```bash
curl -X POST http://localhost:8337/tools/my_tool \
     -H "Content-Type: application/json" \
     -d '{"query": "hello"}'
```

### MCP — Model Context Protocol

When the `[mcp]` extra is installed, the server mounts an MCP SSE endpoint at
`/mcp`.  Each installed `ToolBox` tool is registered as an MCP tool with the
name, description, and JSON Schema derived from its OPM definition.

**Claude Desktop / MCP client config:**

```json
{
  "mcpServers": {
    "ovos-persona-tools": {
      "url": "http://localhost:8337/mcp/sse"
    }
  }
}
```

**Standalone stdio MCP server** (for clients that spawn a subprocess):

```bash
ovos-persona-tools-mcp
```

This runs the same tool set over the stdio MCP transport.

### Writing a ToolBox plugin

Implement `ToolBox` from `ovos_plugin_manager.templates.agent_tools` and
register it under the `opm.agents.toolbox` entry-point group:

```toml
# pyproject.toml
[project.entry-points."opm.agents.toolbox"]
my_toolbox = "my_package.toolbox:MyToolBox"
```

The server picks it up automatically on the next start.

## Client side usage

The OpenAI and Ollama routers expose `/embeddings` endpoints. These require a solver plugin that implements `get_embeddings(text)`. If no such solver is loaded the endpoint returns HTTP 501. See [docs/embeddings.md](docs/embeddings.md).

---

## Authentication

The server itself does not enforce authentication — deploy behind a reverse proxy (nginx, Caddy, Traefik) with TLS and auth if public exposure is required. For the A2A endpoint, A2A clients that require bearer tokens can be configured on the client side (`api_key` in `ovos-a2a-agent` config).

---

## Troubleshooting

**`Failed to load persona` (500 on startup)**
The persona JSON file was not found or is invalid. Check the `--persona` path and validate the JSON.

**`404 model_not_found` on chat requests**
Several personas are loaded and the `model` you sent is not one of them. `GET /openai/v1/models` lists the valid names. See [docs/multi-persona.md](docs/multi-persona.md).

**`duplicate persona name` on startup**
Two loaded persona files declare the same `name`. Names are the model ids and must be unique.

**All requests return `500 Persona chat failed`**
The underlying solver chain failed. Check solver plugin installation and their individual configs (API keys, model paths, etc.).

**A2A endpoint not available after starting with `--a2a-base-url`**
`a2a-sdk` is not installed. Install it:
```bash
uv pip install 'ovos-persona-server[a2a]'
```
Then restart the server.

**Embeddings return 501**
No solver with `get_embeddings()` is loaded and no embeddings plugin could be loaded. Configure `TEXT_EMBEDDINGS_PLUGIN` or add an embeddings solver to the persona's `solvers` list.

**Legacy `/v1/` paths return responses with `Deprecation` header**
This is expected. Migrate to `/openai/v1/` paths. See [docs/deprecation.md](docs/deprecation.md).

## RAG: Files & Vector Stores

`ovos-persona-server` exposes an OpenAI-compatible **Retrieval-Augmented Generation**
surface: upload documents, embed and index them, search by similarity, and feed the
results into any chat endpoint. Files, embedding, and the vector DB are all backed by
swappable OVOS plugins. Full reference: [docs/rag.md](docs/rag.md); runnable scripts in
[examples/](examples/).

> **Drop-in OpenAI replacement:** any third-party app built on OpenAI's Files /
> Vector Stores / Embeddings endpoints can point at this server by changing only its
> `base_url` — a self-hosted, private, zero-cost RAG backend with no code changes.

```bash
uv pip install 'ovos-persona-server[rag]' ovos-gguf-plugin ovos-chromadb-embeddings-plugin

TEXT_EMBEDDINGS_PLUGIN=ovos-gguf-embeddings-plugin EMBEDDINGS_MODEL=all-MiniLM-L6-v2 \
EMBEDDINGS_DB_PLUGIN=ovos-chromadb-embeddings-plugin \
ovos-persona-server --persona examples/persona_rag.json
```

Drive it with the official `openai` SDK — files hit `/openai/v1/files`, the store is a
collection in the configured vector DB:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8337/openai/v1", api_key="unused")

f = client.files.create(file=("cats.txt", b"cats are fluffy animals that sit on mats."),
                        purpose="assistants")
store = client.vector_stores.create(name="kb")
client.vector_stores.files.create(vector_store_id=store.id, file_id=f.id)

hits = client.vector_stores.search(vector_store_id=store.id, query="fluffy animal", max_num_results=3)
print([(h.file_id, h.score) for h in hits.data])
```

For conversational use, the companion [`ovos-openai-plugin`](https://github.com/OpenVoiceOS/ovos-openai-plugin)
ships `PersonaServerRAGMemory` — a persona **memory plugin** that searches a vector
store and injects the retrieved context, composing with any chat backend. See
[docs/rag.md](docs/rag.md#using-rag-as-a-persona-memory-plugin) and
[examples/rag_memory_plugin.py](examples/rag_memory_plugin.py).

> **Backend vs hosted agent.** By default the chat endpoints are a **stateless
> backend** (`CHAT_MEMORY=off`): the client owns conversation state and drives the
> Files / Vector-Stores endpoints itself — the correct behaviour for a drop-in
> OpenAI replacement or any multi-user deployment. Set `CHAT_MEMORY=transparent` to
> run a **single-user hosted agent** where the server folds the persona's
> `memory_module` (history + RAG) into every turn and persists it per session. See
> [docs/rag.md](docs/rag.md#backend-vs-hosted-agent--the-chat_memory-toggle).

| Resource | Endpoints |
| --- | --- |
| Files | `POST/GET /openai/v1/files`, `GET …/{id}`, `GET …/{id}/content`, `DELETE …/{id}` |
| Vector stores | `POST/GET /openai/v1/vector_stores`, `GET/POST/DELETE …/{id}`, `…/{id}/files`, `…/{id}/search` |

---

## Embeddings

A single, swappable embeddings service backs **every** vendor surface — OpenAI
(`/openai/v1/embeddings`), Ollama (`/ollama/api/embed` and `/embeddings`), Cohere
(`/cohere/v1/embed`), Gemini (`:embedContent` / `:batchEmbedContents`), HuggingFace
TGI (`/tgi/embed`), AWS Bedrock (Titan/Cohere embed models via `/invoke`), and the
vector-store search path all delegate to the same backend. (Anthropic has no
first-party embeddings API, so it has no embed endpoint.) This mirrors how inference
is backed by one shared persona: swap the embeddings provider once and it changes
everywhere. Per-surface request/response shapes: [docs/embeddings.md](docs/embeddings.md).

The backend is any OVOS text-embeddings plugin, configured through the
environment:

| Variable | Purpose | Default |
| --- | --- | --- |
| `TEXT_EMBEDDINGS_PLUGIN` | embeddings plugin to load | `ovos-gguf-embeddings-plugin` |
| `EMBEDDINGS_URL` | remote embeddings service URL (OpenAI-compatible plugins) | — |
| `EMBEDDINGS_KEY` | API key for a remote embeddings service | — |
| `EMBEDDINGS_MODEL` | model name to request | — |

Point `TEXT_EMBEDDINGS_PLUGIN` at a local model (the default gguf plugin) or at
any remote embeddings API via an OpenAI-compatible plugin and the matching
`EMBEDDINGS_URL` / `EMBEDDINGS_MODEL`. When no embeddings plugin is available the
server falls back to a persona solver exposing `get_embeddings`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8337/openai/v1", api_key="")
resp = client.embeddings.create(model="", input=["hello", "world"])
print(len(resp.data), "vectors")
```

```python
from ollama import Client

client = Client(host="http://localhost:8337/ollama")
print(client.embed(model="", input=["hello", "world"]).embeddings)
```

---

## Credits

Developed by [TigreGótico](https://tigregotico.pt) for
[OpenVoiceOS](https://openvoiceos.org).

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
