# Quick Facts — `ovos-persona-server`

OpenAI/Ollama-compatible FastAPI server that exposes an OVOS `Persona` instance as an HTTP API.

| Feature | Details |
|---------|---------|
| Package Name | `ovos-persona-server` |
| Version | `0.5.1a2` |
| License | Apache-2.0 |
| Repository | https://github.com/OpenVoiceOS/ovos-persona-server |
| Python Support | >=3.9 |

## Key Classes / Functions

| Symbol | File | Description |
|--------|------|-------------|
| `create_persona_app(persona_path)` | `ovos_persona_server/__init__.py:20` | Loads persona JSON, creates FastAPI app, registers routers |
| `get_default_persona()` | `ovos_persona_server/persona.py` | FastAPI dependency; returns loaded `Persona` or raises HTTP 500 |
| `chat_router` | `ovos_persona_server/chat.py` | OpenAI `/v1` endpoints |
| `ollama_router` | `ovos_persona_server/ollama.py` | Ollama `/api` endpoints |
| `main()` | `ovos_persona_server/__main__.py` | CLI entry point (argparse + uvicorn) |

## Entry Points

| Type | Name | Target |
|------|------|--------|
| Script | `ovos-persona-server` | `ovos_persona_server.__main__:main` |

## API Endpoints

| Method | Path | Protocol |
|--------|------|----------|
| POST | `/v1/chat/completions` | OpenAI Chat Completions |
| POST | `/v1/completions` | OpenAI Legacy Completions |
| GET | `/v1/models` | OpenAI model listing |
| POST | `/api/chat` | Ollama Chat |
| POST | `/api/generate` | Ollama Generate |
| GET | `/api/tags` | Ollama model listing |

## Dependencies

- `fastapi` — HTTP framework
- `ovos-persona` — `Persona` class and solver chain
- `pydantic` — request/response schemas
- `uvicorn` — ASGI server (runtime)
