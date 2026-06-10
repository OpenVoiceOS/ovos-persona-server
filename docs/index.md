# ovos-persona-server — Documentation

## What is this?

`ovos-persona-server` exposes a single OVOS `Persona` as a multi-protocol HTTP API. Eight API surfaces let any LLM client or A2A agent interact with your persona without modification.

## Docs

| File | Contents |
|------|----------|
| [api-compatibility.md](api-compatibility.md) | All 7 non-A2A APIs: prefixes, endpoints, auth, request schemas, curl examples |
| [a2a.md](a2a.md) | A2A endpoint: enabling, Agent Card, streaming, connecting ovos-a2a-agent |
| [streaming.md](streaming.md) | SSE streaming format per API |
| [embeddings.md](embeddings.md) | Embeddings endpoint and solver requirement |
| [bedrock-models.md](bedrock-models.md) | AWS Bedrock model ID detection and response format selection |
| [deprecation.md](deprecation.md) | Legacy `/v1/` and `/api/` paths; migration guide |

## API Surfaces at a Glance

| API | Prefix | Source |
|-----|--------|--------|
| OpenAI | `/openai/v1` | `chat.py` |
| Ollama | `/ollama/api` | `ollama.py` |
| Anthropic | `/anthropic/v1` | `anthropic.py` |
| Google Gemini | `/gemini/v1beta/models` | `gemini.py` |
| Cohere | `/cohere/v1` | `cohere.py` |
| HuggingFace TGI | `/tgi` | `huggingface_tgi.py` |
| AWS Bedrock | `/bedrock/model` | `aws_bedrock.py` |
| A2A | `/a2a` | `a2a.py` (optional) |

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
