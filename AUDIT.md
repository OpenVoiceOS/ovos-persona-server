# ovos-persona-server — Audit Report

_Last updated: 2026-03-17_

## Documentation Status

- [x] `docs/index.md`
- [x] `QUICK_FACTS.md`
- [x] `FAQ.md`
- [x] `MAINTENANCE_REPORT.md`
- [x] `AUDIT.md`
- [x] `SUGGESTIONS.md`

## Known Issues

### Critical

_None._

### Major

- `[MAJOR]` **tests**: No unit tests exist — `ovos_persona_server/__init__.py` and routers have zero test coverage. (`test/` directory absent.)
- `[MAJOR]` **ci**: `publish_stable.yml` and `release_workflow.yml` reference `pypa/gh-action-pypi-publish@master` — should be pinned to `@release/v1`. (`release_workflow.yml`)

### Minor

- `[MINOR]` **ci**: Missing workflows: `lint.yml`, `build_tests.yml`, `license_tests.yml`, `pip_audit.yml` — added 2026-03-17.
- `[MINOR]` **docs**: `pyproject.toml` description still says "simple flask server" — server uses FastAPI, not Flask. (`pyproject.toml:8`)
- `[MINOR]` **code**: `create_persona_app` uses a module-level global (`ovos_persona_server.persona.default_persona`) with a TODO to migrate to dependency injection. (`ovos_persona_server/__init__.py:38`)
- `[MINOR]` **code**: Token counts in `usage` fields use `len(text.split())` — not real tokenizer counts. Documented but worth tracking. (`ovos_persona_server/chat.py`)

### Info

- `[INFO]` **packaging**: `pyproject.toml` `requires-python = ">=3.9"` — workspace standard is >=3.10. Low priority since 3.9 still works.
- `[INFO]` **security**: No authentication on any endpoint. Intentional but should be documented prominently for deployers.

## Technical Debt

| Item | File | Notes |
|------|------|-------|
| Global persona state | `ovos_persona_server/__init__.py:38` | TODO comment present; blocks multi-persona support |
| No unit tests | — | CI has no test or coverage workflow |
| Flask reference in description | `pyproject.toml:8` | Stale copy-paste |
