# Deprecation

## Overview

`ovos-persona-server` previously exposed endpoints at unprefixed paths (`/v1/...` for OpenAI, `/api/...` for Ollama). The canonical paths are now vendor-namespaced:

| Legacy prefix | Canonical prefix |
| :--- | :--- |
| `/v1/...` | `/openai/v1/...` |
| `/api/...` | `/ollama/api/...` |

The legacy paths remain fully functional but emit deprecation signal headers on every response.

**Source**: `ovos_persona_server/deprecated_routers.py` — `register_deprecated_routes()`, `add_deprecation_middleware()`

---

## Deprecation Headers

Every response from a legacy path receives two additional headers:

```
Deprecation: true
Link: </openai/v1/chat/completions>; rel="successor-version"
```

The `Link` header value is the canonical equivalent of the requested path. For example:

| Requested path | Link header value |
| :--- | :--- |
| `/v1/chat/completions` | `</openai/v1/chat/completions>; rel="successor-version"` |
| `/v1/models` | `</openai/v1/models>; rel="successor-version"` |
| `/api/chat` | `</ollama/api/chat>; rel="successor-version"` |
| `/api/tags` | `</ollama/api/tags>; rel="successor-version"` |

A `WARNING`-level log entry is also emitted server-side: `Deprecated path accessed: /v1/... — use /openai/v1/... instead`.

---

## Implementation

`register_deprecated_routes(app)` — `deprecated_routers.py:85`

Clones every route from `chat_router` and `ollama_router` under the legacy prefixes using `app.include_router()`. Because routes are cloned (not redirected), FastAPI's full dependency injection (`Depends`) continues to work on legacy paths. The clone sets `deprecated=True` in the OpenAPI schema.

`add_deprecation_middleware(app)` — `deprecated_routers.py:52`

Attaches an HTTP middleware that inspects `request.url.path`. If the path starts with a known legacy prefix, the middleware injects `Deprecation` and `Link` headers into the response after the handler completes.

`_build_successor_path(request_path)` — `deprecated_routers.py:36`

Computes the canonical path from a legacy path by prefix substitution.

---

## Migration Guide

Replace only the path prefix in your client configuration:

| Before | After |
| :--- | :--- |
| `http://host:8337/v1/chat/completions` | `http://host:8337/openai/v1/chat/completions` |
| `http://host:8337/v1/models` | `http://host:8337/openai/v1/models` |
| `http://host:8337/v1/embeddings` | `http://host:8337/openai/v1/embeddings` |
| `http://host:8337/api/chat` | `http://host:8337/ollama/api/chat` |
| `http://host:8337/api/generate` | `http://host:8337/ollama/api/generate` |
| `http://host:8337/api/tags` | `http://host:8337/ollama/api/tags` |

No request body or header changes are required — the canonical endpoints accept the same schemas.

**Open WebUI**: Change the API base URL in Settings → Connections from `http://host:8337` to `http://host:8337/openai` (for OpenAI mode) or update the Ollama URL to `http://host:8337/ollama`.

**LangChain `ChatOpenAI`**: Set `openai_api_base="http://host:8337/openai/v1"`.

**Ollama CLI**: Set `OLLAMA_HOST=http://host:8337/ollama`.
