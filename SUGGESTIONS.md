# Suggestions — `ovos-persona-server`

## 1. Fix lifespan placement

**Problem**: `chat_router` and `ollama_router` pass a `lifespan` parameter to `APIRouter()` (`chat.py:30`, `ollama.py:36`). FastAPI ignores `lifespan` on `APIRouter`; it is only honoured on `FastAPI()`. The default persona is not guaranteed to be warmed up before the first request.

**Proposed Solution**: Move lifespan logic to `create_persona_app()` in `__init__.py` using `@asynccontextmanager` on the `FastAPI` app, or rely on the existing `get_default_persona()` dependency which initialises on first use.

**Estimated Impact**: Low effort; removes silent no-op code.

---

## 2. Replace global persona state with dependency injection

**Problem**: `default_persona` is a module-level global (`ovos_persona_server/__init__.py:38`). A TODO comment acknowledges this. Prevents multi-persona support and makes test isolation harder.

**Proposed Solution**: Use `app.state.persona` set in the FastAPI lifespan and inject via `request.app.state` or a closure-based dependency.

**Estimated Impact**: Medium effort; enables multi-persona and cleaner tests.

---

## 3. Real token counting

**Problem**: `usage.prompt_tokens` / `completion_tokens` across all routers use `len(text.split())`. Clients tracking cost or enforcing context limits get wrong values.

**Proposed Solution**: Use `tiktoken` for OpenAI responses, or document explicitly that counts are approximate word splits. A config flag `approximate_tokens: true` could make this opt-in.

**Estimated Impact**: Low effort; improves compatibility with strict clients.

---

## 4. Fix stale pyproject.toml description

**Problem**: `description = "simple flask server ..."` — the server uses FastAPI. (`pyproject.toml:8`)

**Proposed Solution**: `"Multi-protocol FastAPI server exposing an OVOS Persona as OpenAI/Ollama/Anthropic/Gemini/Cohere/TGI/Bedrock-compatible APIs."`

**Estimated Impact**: Cosmetic; improves PyPI listing.

---

## 5. Add Bedrock `/invoke` body disambiguation by model_id

**Problem**: `_extract_messages` detects format by body field presence, not `model_id`. If a request to an `amazon.titan` model accidentally includes a `"messages"` key, it will be parsed as Anthropic format. (`aws_bedrock.py:38`)

**Proposed Solution**: Check `model_id` prefix first; fall back to body field heuristics only for unknown prefixes.

**Estimated Impact**: Low effort; prevents silent misparse.

---

## 6. Add coverage CI workflow

**Problem**: No coverage workflow is connected to the existing 31 tests.

**Proposed Solution**: Add `coverage.yml` from `OpenVoiceOS/gh-automations@dev` once pyproject.toml is confirmed compatible.

**Estimated Impact**: Low effort; enables CI enforcement.
