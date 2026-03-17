# Suggestions — `ovos-persona-server`

## 1. Replace global persona state with dependency injection

**Problem**: `default_persona` is a module-level global set in `create_persona_app()` (`ovos_persona_server/__init__.py:38`). A TODO comment acknowledges this. It prevents serving multiple personas from one process and makes testing harder.

**Proposed Solution**: Use FastAPI's `Depends()` with a lifespan-scoped `app.state.persona`. Pass it via `request.app.state` or a closure-based dependency.

**Estimated Impact**: Medium effort; enables multi-persona support and cleaner unit tests.

---

## 2. Add unit tests

**Problem**: No `test/` directory exists. CI has no test or coverage workflow. Any refactor is unverified.

**Proposed Solution**: Create `test/unittests/test_app.py` using `fastapi.testclient.TestClient`. Test: app creation, `/v1/chat/completions` (non-streaming and streaming), `/api/chat`, `/api/tags`, error path when persona fails.

**Estimated Impact**: High — enables CI coverage enforcement and safe refactoring.

---

## 3. Add `test.yml` and `coverage.yml` workflows

**Problem**: No test or coverage CI workflows are present.

**Proposed Solution**: Add `test.yml` and `coverage.yml` from `OpenVoiceOS/gh-automations@dev` once unit tests exist.

**Estimated Impact**: Low effort after suggestion #2 is complete.

---

## 4. Fix stale description in `pyproject.toml`

**Problem**: `description = "simple flask server ..."` — the server uses FastAPI, not Flask. (`pyproject.toml:8`)

**Proposed Solution**: Update to `"OpenAI/Ollama-compatible FastAPI server for OVOS Personas and Solvers"`.

**Estimated Impact**: Cosmetic; improves PyPI listing accuracy.

---

## 5. Real token counting

**Problem**: `usage.prompt_tokens` / `completion_tokens` use `len(text.split())`. Clients that rely on accurate token counts (e.g. for cost tracking) will get wrong values.

**Proposed Solution**: Use `tiktoken` (or a lightweight BPE approximation) for token counting, or document clearly that counts are approximate.

**Estimated Impact**: Low effort; improves compatibility with strict OpenAI clients.
