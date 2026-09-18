# Embeddings

## Overview

A **single, swappable embeddings backend** powers every embeddings surface and the
vector-store search path. Each endpoint only translates request/response shapes; they
all delegate to the same `get_embeddings_backend` (`ovos_persona_server/embeddings.py`).
Swap the provider in one place (`TEXT_EMBEDDINGS_PLUGIN`) and it changes everywhere.

| Endpoint | Router | Shape |
| :--- | :--- | :--- |
| `POST /openai/v1/embeddings` | `chat.py` | OpenAI |
| `POST /ollama/api/embed` | `ollama.py` | Ollama batch (`input`) |
| `POST /ollama/api/embeddings` | `ollama.py` | Ollama legacy (`prompt`) |
| `POST /cohere/v1/embed` | `cohere.py` | Cohere |
| `POST /gemini/v1beta/models/{model}:embedContent` | `gemini.py` | Gemini (single) |
| `POST /gemini/v1beta/models/{model}:batchEmbedContents` | `gemini.py` | Gemini (batch) |
| `POST /tgi/embed` | `huggingface_tgi.py` | HF Text-Embeddings-Inference |
| `POST /bedrock/model/{model}/invoke` | `aws_bedrock.py` | Bedrock Titan / Cohere embed |

Anthropic has no first-party embeddings API, so the Anthropic surface intentionally
exposes none.

## Backend resolution

`get_embeddings_backend` resolves, in order:

1. the configured `TEXT_EMBEDDINGS_PLUGIN` (default `ovos-gguf-embeddings-plugin`) —
   point it at a remote service with `EMBEDDINGS_URL` / `EMBEDDINGS_MODEL` /
   `EMBEDDINGS_KEY` for OpenAI-compatible plugins;
2. otherwise, a persona solver exposing `get_embeddings(text) -> list[float]`.

If neither is available the endpoints return **HTTP 501 Not Implemented**:

```json
{"detail": "No embeddings backend available: configure TEXT_EMBEDDINGS_PLUGIN ..."}
```

See [rag.md](rag.md) for how the same backend drives vector-store search.

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

### Gemini (`POST /gemini/v1beta/models/{model}:embedContent`)

**Request**: `{"content": {"parts": [{"text": "text to embed"}]}}`
**Response**: `{"embedding": {"values": [0.1, 0.2, ...]}}`

The batch variant `:batchEmbedContents` takes `{"requests": [{"content": {...}}, ...]}`
and returns `{"embeddings": [{"values": [...]}, ...]}`.

### HuggingFace TGI (`POST /tgi/embed`)

Text-Embeddings-Inference shape — **Request**: `{"inputs": "text" | ["a", "b"]}`,
**Response**: a bare JSON array of vectors `[[0.1, ...], ...]`.

### AWS Bedrock (`POST /bedrock/model/{model}/invoke`)

Embedding model ids (`amazon.titan-embed-*`, `cohere.embed-*`) route to the backend:
- Titan — **Request** `{"inputText": "..."}` → **Response** `{"embedding": [...], "inputTextTokenCount": N}`
- Cohere — **Request** `{"texts": [...]}` → **Response** `{"embeddings": [[...]], "texts": [...], ...}`

## Notes

- Token counts in `usage` are always zero — the backend does not report them.
- `encoding_format` and `dimensions` from the OpenAI request schema are accepted but not acted on; the vector format is determined entirely by the backend (`encoding_format="base64"` returns 500 — not yet implemented).
