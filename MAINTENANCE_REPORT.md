# Maintenance Report — `ovos-persona-server`

## [2026-03-18b] — Add Ollama generate + embeddings tests (39 → 41)

### Changes

- Added `TestOllamaRouter::test_generate_returns_done_true` — verifies `/ollama/api/generate` returns 200 with `message` and `done: true`.
- Added `TestOllamaRouter::test_embeddings_no_solver_returns_501` — verifies `/ollama/api/embeddings` returns 501 when no solver has `get_embeddings`.
- Updated `QUICK_FACTS.md` — test count 31 → 41 (reflects all tests across both sessions).

### AI Transparency Report

- **AI Model**: claude-sonnet-4-6
- **Actions Taken**: Added 2 tests; updated QUICK_FACTS.md.
- **Oversight**: Human review required. Tests verified via `uv run pytest test/ -v` (41 passed).

---

## [2026-03-18] — Add 7-API docs; extend test suite with 10 new Ollama/Cohere/TGI/Bedrock tests

### Changes

- Rewrote `docs/index.md` — updated table of contents and prefix table for all 7 APIs
- Created `docs/api-compatibility.md` — all 7 APIs: prefixes, endpoints, auth, request schemas, curl examples
- Created `docs/deprecation.md` — legacy path mechanism, headers, migration guide
- Created `docs/streaming.md` — SSE format documented per API with example payloads
- Created `docs/embeddings.md` — embeddings endpoint behavior, 3 routers, solver requirement, request/response schemas
- Created `docs/bedrock-models.md` — `_extract_messages` and `_build_response` detection logic documented
- Rewrote `FAQ.md` — 20+ Q&As covering all 7 APIs, deprecation, streaming, embeddings, auth
- Updated `QUICK_FACTS.md` — full endpoint table for all 25 canonical paths + 2 legacy; all 11 key symbols
- Updated `AUDIT.md` — added router lifespan issue and Bedrock parsing ambiguity; removed resolved test coverage issue
- Updated `SUGGESTIONS.md` — added lifespan fix and Bedrock disambiguation suggestions
- Added 10 new unit tests to `test/unittests/test_compat_routers.py` — Ollama (5), Cohere generate, TGI info, Bedrock converse, deprecated `/api/...` paths (2)

### Rationale

Previous documentation covered only OpenAI and Ollama prefixes. The repo had grown to 7 APIs and 31 tests with no corresponding documentation update.

### AI Transparency Report

- **AI Model**: Claude Sonnet 4.6
- **Actions Taken**: Created 5 new docs files, updated 5 existing files, extended test suite with 10 tests.
- **Oversight**: Human review required before merging. Tests must pass via `uv run pytest test/ -v`.

---

## [2026-03-17] — Add missing workflows; enrich documentation

### Changes

- Added `.github/workflows/lint.yml` — calls `gh-automations/lint.yml@dev`
- Added `.github/workflows/build_tests.yml` — calls `gh-automations/build-tests.yml@dev`
- Added `.github/workflows/license_tests.yml` — calls `gh-automations/license-check.yml@dev`
- Added `.github/workflows/pip_audit.yml` — calls `gh-automations/pip-audit.yml@dev`
- Rewrote `QUICK_FACTS.md` — accurate version, endpoints table, key class/file citations
- Rewrote `FAQ.md` — accurate API descriptions, streaming behaviour, schema locations
- Rewrote `AUDIT.md` — evidence-based issues with file:line citations
- Rewrote `SUGGESTIONS.md` — repo-specific proposals with file:line citations

### Rationale

Prior documentation stubs contained generic placeholder content and inaccurate references (Flask vs FastAPI). CI was missing four standard OVOS workflows.

### Verification

- Workflow files validated against `gh-automations` reusable workflow names.
- Documentation cross-checked against `ovos_persona_server/__init__.py` source.

### AI Transparency Report

- **AI Model**: Claude Sonnet 4.6
- **Actions Taken**: Created 4 workflow files; rewrote 4 documentation files with source-cited content.
- **Oversight**: Human review required before treating documentation as authoritative.

---

## [2026-03-08] — Initial compliance scaffold

### Changes

- Created `QUICK_FACTS.md`, `FAQ.md`, `MAINTENANCE_REPORT.md`, `SUGGESTIONS.md`, `docs/index.md`, `AUDIT.md` as compliance stubs.

### AI Transparency Report

- **AI Model**: Claude Sonnet 4.6
- **Actions Taken**: Generated boilerplate compliance scaffold.
- **Oversight**: Files were stubs — content enriched 2026-03-17.
