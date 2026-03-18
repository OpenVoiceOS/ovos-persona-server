# ovos-persona-server

OVOS Persona Server exposes a single OVOS `Persona` instance as a multi-protocol HTTP API. Seven canonical router prefixes faithfully replicate each upstream provider's wire format.

## Table of Contents

- [API Compatibility](api-compatibility.md) — all 7 APIs: prefixes, endpoints, auth, request schemas, curl examples
- [Deprecation](deprecation.md) — legacy `/v1/` and `/api/` paths and migration guide
- [Streaming](streaming.md) — SSE format per API
- [Embeddings](embeddings.md) — embeddings endpoint and solver requirement
- [Bedrock Models](bedrock-models.md) — model_id prefix detection and response format selection

## Canonical API Prefixes

| Router | Prefix | Source file |
| :--- | :--- | :--- |
| OpenAI | `/openai/v1` | `ovos_persona_server/chat.py` |
| Ollama | `/ollama/api` | `ovos_persona_server/ollama.py` |
| Anthropic | `/anthropic/v1` | `ovos_persona_server/anthropic.py` |
| Gemini | `/gemini/v1beta/models` | `ovos_persona_server/gemini.py` |
| Cohere | `/cohere/v1` | `ovos_persona_server/cohere.py` |
| HuggingFace TGI | `/tgi` | `ovos_persona_server/huggingface_tgi.py` |
| AWS Bedrock | `/bedrock/model` | `ovos_persona_server/aws_bedrock.py` |

## Deprecated Legacy Paths

| Legacy prefix | Canonical prefix |
| :--- | :--- |
| `/v1/...` | `/openai/v1/...` |
| `/api/...` | `/ollama/api/...` |

See [deprecation.md](deprecation.md) for migration details.

## Architecture

- `ovos_persona_server/persona.py` — `get_default_persona()` FastAPI dependency
- `ovos_persona_server/deprecated_routers.py` — `register_deprecated_routes()`, `add_deprecation_middleware()`
- `ovos_persona_server/__main__.py` — Uvicorn entry point
- `ovos_persona_server/schemas/` — Pydantic request/response models
