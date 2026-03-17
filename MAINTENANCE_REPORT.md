# Maintenance Report — `ovos-persona-server`

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
