# ovos-persona-server — Documentation

## What is this?

`ovos-persona-server` exposes a single OVOS `Persona` as a multi-protocol HTTP API. Eight API surfaces let any LLM client or A2A agent interact with your persona without modification.

## Docs

| File | Contents |
|------|----------|
| [api-compatibility.md](api-compatibility.md) | All 7 non-A2A APIs: prefixes, endpoints, auth, request schemas, curl examples |
| [streaming.md](streaming.md) | SSE streaming format per API |
| [rag.md](rag.md) | Files, Vector Stores, and the RAG flow (upload → embed → search → chat) |
| [embeddings.md](embeddings.md) | Shared embeddings backend across every vendor surface |
| [deprecation.md](deprecation.md) | Legacy `/v1/` and `/api/` paths; migration guide |

Runnable examples live in [`examples/`](../examples/).

## API Surfaces at a Glance

| API | Prefix | Source | Docs |
|-----|--------|--------|------|
| OpenAI | `/openai/v1` | `chat.py` | [api-compatibility.md](api-compatibility.md#1-openai----openaiv1) |
| OpenAI RAG (Files + Vector Stores) | `/openai/v1/files`, `/openai/v1/vector_stores` | `files.py`, `vector_stores.py` | [rag.md](rag.md) |
| Ollama | `/ollama/api` | `ollama.py` | [api-compatibility.md](api-compatibility.md#2-ollama----ollamaapi) |
| Anthropic | `/anthropic/v1` | `anthropic.py` | [api-compatibility.md](api-compatibility.md#3-anthropic----anthropicv1) |
| Google Gemini | `/gemini/v1beta/models` | `gemini.py` | [api-compatibility.md](api-compatibility.md#4-gemini----geminiv1betamodels) |
| Cohere | `/cohere/v1` | `cohere.py` | [api-compatibility.md](api-compatibility.md#5-cohere----coherev1) |
| HuggingFace TGI | `/tgi` | `huggingface_tgi.py` | [api-compatibility.md](api-compatibility.md#6-huggingface-tgi----tgi) |
| AWS Bedrock | `/bedrock/model` | `aws_bedrock.py` | [api-compatibility.md](api-compatibility.md#7-aws-bedrock----bedrockmodel) |
| A2A | `/a2a` | `a2a.py` (optional) | see `a2a.py` module docstring |

## Architecture

```
HTTP request
  └─ FastAPI app  [__init__.py:create_persona_app]
       ├─ chat_router          /openai/v1/...
       ├─ ollama_router        /ollama/api/...
       ├─ anthropic_router     /anthropic/v1/...
       ├─ gemini_router        /gemini/v1beta/...
       ├─ cohere_router        /cohere/v1/...
       ├─ tgi_router           /tgi/...
       ├─ bedrock_router       /bedrock/model/...
       ├─ deprecated routes    /v1/... /api/... (with Deprecation header)
       └─ A2A Starlette app    /a2a/... (optional; requires a2a-sdk)
            └─ OVOSPersonaAgentExecutor  [a2a.py]
                 └─ Persona.stream()     [ovos-persona]
```

## Key Classes

| Symbol | File | Role |
|--------|------|------|
| `create_persona_app` | `__init__.py:20` | Factory: loads persona, wires all routers |
| `get_default_persona` | `persona.py:17` | FastAPI dependency; returns loaded `Persona` |
| `OVOSPersonaAgentExecutor` | `a2a.py:106` | A2A `AgentExecutor` wrapping `Persona.stream()` |
| `_agent_card` | `a2a.py:66` | Builds A2A `AgentCard` from persona metadata |
| `create_a2a_application` | `a2a.py:212` | Returns `A2AStarletteApplication` for mounting |
