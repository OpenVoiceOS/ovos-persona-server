# Deprecation

## Overview

The canonical paths of `ovos-persona-server` are vendor-namespaced. The unprefixed paths (`/v1/...` for OpenAI, `/api/...` for Ollama) are deprecated. Use the canonical prefix:

| Legacy prefix | Canonical prefix |
| :--- | :--- |
| `/v1/...` | `/openai/v1/...` |
| `/api/...` | `/ollama/api/...` |

The legacy paths remain fully functional but emit deprecation signal headers on every response.

**Source**: `ovos_persona_server/deprecated_routers.py`, functions `register_deprecated_routes()`, and `add_deprecation_middleware()`

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

The server also writes a `WARNING`-level log entry that starts with `Deprecated path accessed:` and names the legacy path and the canonical path.

---

## Implementation

`register_deprecated_routes(app)` is in `deprecated_routers.py:85`.

Clones every route from `chat_router` and `ollama_router` under the legacy prefixes using `app.include_router()`. Because routes are cloned (not redirected), FastAPI's full dependency injection (`Depends`) continues to work on legacy paths. The clone sets `deprecated=True` in the OpenAPI schema.

`add_deprecation_middleware(app)` is in `deprecated_routers.py:52`.

Attaches an HTTP middleware that inspects `request.url.path`. If the path starts with a known legacy prefix, the middleware injects `Deprecation` and `Link` headers into the response after the handler completes.

`_build_successor_path(request_path)` is in `deprecated_routers.py:36`.

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

No request body or header change is required. The canonical endpoints accept the same schemas.

**Open WebUI**: Change the API base URL in Settings → Connections from `http://host:8337` to `http://host:8337/openai` (for OpenAI mode) or update the Ollama URL to `http://host:8337/ollama`.

**LangChain `ChatOpenAI`**: Set `openai_api_base="http://host:8337/openai/v1"`.

**Ollama CLI**: Set `OLLAMA_HOST=http://host:8337/ollama`.
