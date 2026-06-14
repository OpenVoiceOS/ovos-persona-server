# ovos-persona-server

A single HTTP server that exposes one OVOS `Persona` as **eight concurrent API surfaces** — so any LLM client (OpenAI SDK, LangChain, Ollama tools, Anthropic SDK, Google Gemini SDK, Cohere SDK, HuggingFace TGI client, AWS Bedrock client, or any A2A agent) can talk to your OVOS persona without changes.

---

## Table of Contents

- [What is a Persona?](#what-is-a-persona)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Surfaces](#api-surfaces)
- [A2A Endpoint](#a2a-endpoint)
- [Persona Config Examples](#persona-config-examples)
- [Streaming](#streaming)
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

## API Surfaces

Every API is served on a vendor-prefixed path so multiple clients can coexist without conflict.

| API | Prefix | Key endpoints |
|-----|--------|---------------|
| OpenAI | `/openai/v1` | `POST /chat/completions`, `POST /completions`, `GET /models`, `POST /embeddings` |
| Ollama | `/ollama/api` | `POST /chat`, `POST /generate`, `GET /tags`, `POST /embeddings` |
| Anthropic | `/anthropic/v1` | `POST /messages` |
| Google Gemini | `/gemini/v1beta/models` | `POST /{model}:generateContent`, `POST /{model}:streamGenerateContent` |
| Cohere | `/cohere/v1` | `POST /chat` |
| HuggingFace TGI | `/tgi` | `POST /generate`, `POST /generate_stream` |
| AWS Bedrock | `/bedrock/model` | `POST /{model}/invoke`, `POST /{model}/invoke-with-response-stream` |
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

**All requests return `500 Persona chat failed`**
The underlying solver chain failed. Check solver plugin installation and their individual configs (API keys, model paths, etc.).

**A2A endpoint not available after starting with `--a2a-base-url`**
`a2a-sdk` is not installed. Install it:
```bash
uv pip install 'ovos-persona-server[a2a]'
```
Then restart the server.

**Embeddings return 501**
No solver with `get_embeddings()` is loaded. Add an embeddings solver to the persona's `solvers` list.

**Legacy `/v1/` paths return responses with `Deprecation` header**
This is expected. Migrate to `/openai/v1/` paths. See [docs/deprecation.md](docs/deprecation.md).

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
