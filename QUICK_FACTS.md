# Quick Facts — `ovos-persona-server`

Multi-protocol FastAPI server that exposes a single OVOS `Persona` instance across seven API surfaces.

| Feature | Details |
| :--- | :--- |
| Package | `ovos-persona-server` |
| Version | `0.5.1a2` |
| License | Apache-2.0 |
| Repository | https://github.com/OpenVoiceOS/ovos-persona-server |
| Python | >= 3.9 |

## Key Classes / Functions

| Symbol | File | Description |
| :--- | :--- | :--- |
| `create_persona_app(persona_path)` | `ovos_persona_server/__init__.py:20` | Loads persona JSON, creates FastAPI app, registers all routers |
| `get_default_persona()` | `ovos_persona_server/persona.py` | FastAPI dependency returning loaded `Persona` or HTTP 500 |
| `chat_router` | `ovos_persona_server/chat.py:40` | OpenAI `/openai/v1` endpoints |
| `ollama_router` | `ovos_persona_server/ollama.py:47` | Ollama `/ollama/api` endpoints |
| `anthropic_router` | `ovos_persona_server/anthropic.py:21` | Anthropic `/anthropic/v1` endpoints |
| `gemini_router` | `ovos_persona_server/gemini.py:19` | Gemini `/gemini/v1beta/models` endpoints |
| `cohere_router` | `ovos_persona_server/cohere.py:16` | Cohere `/cohere/v1` endpoints |
| `tgi_router` | `ovos_persona_server/huggingface_tgi.py:13` | TGI `/tgi` endpoints |
| `bedrock_router` | `ovos_persona_server/aws_bedrock.py:16` | Bedrock `/bedrock/model` endpoints |
| `register_deprecated_routes(app)` | `ovos_persona_server/deprecated_routers.py:85` | Mounts `/v1/` and `/api/` legacy aliases |
| `add_deprecation_middleware(app)` | `ovos_persona_server/deprecated_routers.py:52` | Injects `Deprecation` + `Link` headers on legacy paths |
| `main()` | `ovos_persona_server/__main__.py` | CLI entry point (argparse + uvicorn) |

## API Endpoints

| Method | Canonical Path | Protocol |
| :--- | :--- | :--- |
| POST | `/openai/v1/chat/completions` | OpenAI Chat (streaming) |
| POST | `/openai/v1/completions` | OpenAI Legacy Completions (streaming) |
| GET | `/openai/v1/models` | OpenAI model listing |
| POST | `/openai/v1/embeddings` | OpenAI embeddings (requires solver) |
| POST | `/ollama/api/chat` | Ollama Chat (streaming) |
| POST | `/ollama/api/generate` | Ollama Generate (streaming) |
| GET | `/ollama/api/tags` | Ollama model listing |
| GET | `/ollama/api/show` | Ollama model card |
| GET | `/ollama/api/ps` | Ollama running models |
| POST | `/ollama/api/pull` | Ollama pull (stub) |
| POST | `/ollama/api/push` | Ollama push (stub) |
| POST | `/ollama/api/embeddings` | Ollama embeddings (requires solver) |
| POST | `/anthropic/v1/messages` | Anthropic messages (streaming) |
| POST | `/gemini/v1beta/models/{id}:generateContent` | Gemini generate |
| POST | `/gemini/v1beta/models/{id}:streamGenerateContent` | Gemini stream |
| POST | `/cohere/v1/chat` | Cohere chat (streaming) |
| POST | `/cohere/v1/generate` | Cohere generate (streaming) |
| POST | `/cohere/v1/embed` | Cohere embeddings (requires solver) |
| POST | `/tgi/generate` | TGI generate |
| POST | `/tgi/generate_stream` | TGI generate stream |
| GET | `/tgi/info` | TGI model info |
| GET | `/tgi/health` | TGI health check |
| POST | `/bedrock/model/{id}/invoke` | Bedrock invoke |
| POST | `/bedrock/model/{id}/invoke-with-response-stream` | Bedrock stream |
| POST | `/bedrock/model/{id}/converse` | Bedrock converse |

Legacy aliases: `/v1/...` → `/openai/v1/...`, `/api/...` → `/ollama/api/...` (with `Deprecation: true` header).

## Unit Tests

41 tests in `test/unittests/test_compat_routers.py`. Run: `uv run pytest test/ -v`.

## Dependencies

- `fastapi`, `uvicorn` — HTTP framework and server
- `ovos-persona` — `Persona` class and solver chain
- `ovos-bus-client` — `SessionManager` (Ollama endpoints)
- `pydantic` — request/response schemas
