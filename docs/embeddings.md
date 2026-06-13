# Embeddings

## Overview

Embeddings are supported on three endpoints across two routers. All three delegate to the same mechanism: iterating `persona.solvers.loaded_modules` and finding the first solver with a `get_embeddings` method.

| Endpoint | Router | Source |
| :--- | :--- | :--- |
| `POST /openai/v1/embeddings` | `chat_router` | `ovos_persona_server/chat.py:355` |
| `POST /ollama/api/embeddings` | `ollama_router` | `ovos_persona_server/ollama.py:486` |
| `POST /cohere/v1/embed` | `cohere_router` | `ovos_persona_server/cohere.py:185` |

## Solver Requirement

If no loaded solver exposes `get_embeddings`, all three endpoints return **HTTP 501 Not Implemented**:

```json
{"detail": "No embeddings solver configured for this persona."}
```

To enable embeddings, configure a persona with a solver plugin that implements `get_embeddings(text: str) -> list[float]`.

## Request and Response Formats

### OpenAI (`POST /openai/v1/embeddings`)

**Request** (`OpenAIEmbeddingsRequest` — `ovos_persona_server/chat.py:345`):
```json
{
  "model": "text-embedding-ada-002",
  "input": "text to embed"
}
```
`input` may be a string or list of strings.

**Response**:
```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.1, 0.2, ...], "index": 0}
  ],
  "model": "text-embedding-ada-002",
  "usage": {"prompt_tokens": 0, "total_tokens": 0}
}
```

### Ollama (`POST /ollama/api/embeddings`)

**Request** (`OllamaEmbedRequest` — `ovos_persona_server/schemas/ollama.py`):
```json
{
  "model": "any-string",
  "input": "text to embed"
}
```
`input` may be a string or list of strings. If a list, strings are joined with a space before passing to the solver.

**Response** (`OllamaEmbedResponse`):
```json
{
  "embeddings": [[0.1, 0.2, ...]]
}
```

### Cohere (`POST /cohere/v1/embed`)

**Request** (`CohereEmbedRequest` — `ovos_persona_server/cohere.py:45`):
```json
{
  "texts": ["text one", "text two"],
  "input_type": "search_document"
}
```
`texts` must be a non-empty list.

**Response**:
```json
{
  "id": "<24 chars>",
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "texts": ["text one", "text two"],
  "meta": {"api_version": {"version": "1"}}
}
```

## Notes

- Token counts in `usage` are always zero — the underlying solver does not report them.
- `encoding_format` and `dimensions` from the OpenAI request schema are accepted but not acted on; the vector format is determined entirely by the solver.
