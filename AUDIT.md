# ovos-persona-server — Audit Report

_Last updated: 2026-03-18_

## Documentation Status

- [x] `docs/index.md`
- [x] `docs/api-compatibility.md`
- [x] `docs/deprecation.md`
- [x] `docs/streaming.md`
- [x] `docs/embeddings.md`
- [x] `docs/bedrock-models.md`
- [x] `QUICK_FACTS.md`
- [x] `FAQ.md`
- [x] `MAINTENANCE_REPORT.md`
- [x] `AUDIT.md`
- [x] `SUGGESTIONS.md`

## Known Issues

### Critical

_None._

### Major

- `[MAJOR]` **ci**: `publish_stable.yml` and `release_workflow.yml` reference `pypa/gh-action-pypi-publish@master` — should be `@release/v1`. (`release_workflow.yml`)

### Minor

- `[MINOR]` **code**: `create_persona_app` uses a module-level global (`ovos_persona_server.persona.default_persona`) with a TODO comment. Blocks multi-persona support and makes test isolation harder. (`ovos_persona_server/__init__.py:38`)
- `[MINOR]` **code**: Token counts in `usage` use `len(text.split())` — not real tokenizer counts. Affects OpenAI, Anthropic, Cohere, Bedrock, TGI responses. (`ovos_persona_server/chat.py`, `anthropic.py`, `cohere.py`, `aws_bedrock.py`, `huggingface_tgi.py`)
- `[MINOR]` **docs**: `pyproject.toml` description says "simple flask server" — server uses FastAPI. (`pyproject.toml:8`)
- `[MINOR]` **code**: `ollama.py` uses `asynccontextmanager` `lifespan` on `ollama_router` (line 36) — lifespans on `APIRouter` are not supported in FastAPI; the lifespan silently has no effect when included via `app.include_router()`. (`ovos_persona_server/ollama.py:36`)
- `[MINOR]` **code**: Same lifespan issue in `chat.py:30`. (`ovos_persona_server/chat.py:30`)

### Info

- `[INFO]` **packaging**: `requires-python = ">=3.9"` — workspace standard is >= 3.10.
- `[INFO]` **security**: No authentication on any endpoint. Intentional but should be prominent in deployment docs.
- `[INFO]` **code**: Bedrock `/invoke` body parsing uses field presence heuristics (`"messages"`, `"prompt"`, `"inputText"`, `"message"`) rather than `model_id` prefix. Ambiguous if a body contains multiple keys. (`ovos_persona_server/aws_bedrock.py:38`)
- `[INFO]` **code**: Ollama `/pull` and `/push` are stubs that always return `{"status": "success"}` without downloading or uploading anything. (`ovos_persona_server/ollama.py:460-483`)

## Technical Debt

| Item | File | Notes |
| :--- | :--- | :--- |
| Global persona state | `ovos_persona_server/__init__.py:38` | TODO present; blocks multi-persona and clean test isolation |
| Approximate token counts | `chat.py`, `anthropic.py`, `cohere.py`, `aws_bedrock.py`, `huggingface_tgi.py` | word-split, not tokenizer |
| Flask reference in description | `pyproject.toml:8` | Stale copy-paste |
| Router-level lifespan | `chat.py:30`, `ollama.py:36` | FastAPI ignores lifespan on `APIRouter` |
