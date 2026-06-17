# RAG: Files, Vector Stores & Embeddings

`ovos-persona-server` ships an OpenAI-compatible **Retrieval-Augmented Generation**
surface: upload documents, embed and index them in a vector store, search by
similarity, and feed the results back into any chat endpoint. Every piece is backed
by swappable OVOS plugins, so the same API runs on a local laptop or a cloud box.

- **Files API** — `/openai/v1/files` — upload / list / retrieve / delete documents
- **Vector Stores API** — `/openai/v1/vector_stores` — create stores, attach files
  (chunk + embed), and search
- **Embeddings API** — one shared backend exposed on every vendor surface
  (see [embeddings.md](embeddings.md))

The official `openai` Python SDK works against all of it unmodified.

## Architecture

```
            ┌──────────────── shared, swappable backend ────────────────┐
files ──▶ SQLite metadata (metadata.py)                                  │
            │                                                            │
vector  ──▶ EmbeddingsDB plugin  ◀── text embedder plugin ──────────────┘
store        (chromadb / qdrant)      (gguf / any OVOS TextEmbedder)
                  ▲                          ▲
                  └── /vector_stores/search ─┘  ← same embedder the
                                                  /embeddings endpoints use
```

- **Text embedder** — any OVOS `TextEmbedder` plugin. One instance backs the
  `/embeddings` endpoints **and** vector-store search, so swapping the provider in
  one place changes it everywhere. Falls back to a persona solver exposing
  `get_embeddings` when no plugin is installed.
- **Vector DB** — any OVOS `EmbeddingsDB` plugin (each vector store maps to a
  collection). chromadb and qdrant are tested.
- **File metadata** — a unified async SQLite database (files, vector stores, chunks).

## Configuration

All driven by environment variables (see [`config.py`](../ovos_persona_server/config.py)):

| Variable | Purpose | Default |
| --- | --- | --- |
| `TEXT_EMBEDDINGS_PLUGIN` | text-embeddings plugin to load | `ovos-gguf-embeddings-plugin` |
| `EMBEDDINGS_MODEL` | model the embedder should load | — |
| `EMBEDDINGS_DB_PLUGIN` | vector-store backend plugin | `ovos-chromadb-embeddings-plugin` |
| `EMBEDDINGS_URL` / `EMBEDDINGS_KEY` | remote embeddings service (OpenAI-compatible plugins) | — |
| `FILE_STORAGE_PATH` | directory for the SQLite DB, uploads, and vector DB | `~/.local/share/ovos_persona_server` |

Install the RAG plugin stack:

```bash
uv pip install ovos-persona-server[rag] \
  ovos-gguf-plugin ovos-chromadb-embeddings-plugin
```

The `[rag]` extra pulls the chunker and reranker; the embedder and vector DB are
ordinary OVOS plugins you choose per deployment (here: gguf + chromadb).

## End-to-end with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8337/openai/v1", api_key="unused")

# 1. Upload documents
docs = {
    "cats.txt": b"the cat sat on the mat. cats are fluffy animals that purr.",
    "moon.txt": b"the moon orbits the earth and affects ocean tides.",
}
file_ids = [client.files.create(file=(name, body), purpose="assistants").id
            for name, body in docs.items()]

# 2. Create a vector store and attach the files (chunked + embedded server-side)
store = client.vector_stores.create(name="kb")
for fid in file_ids:
    client.vector_stores.files.create(vector_store_id=store.id, file_id=fid)

# 3. Search by similarity
hits = client.vector_stores.search(
    vector_store_id=store.id,
    query="fluffy animal on a mat",
    max_num_results=3,
)
for r in hits.data:
    print(r.score, r.file_id)
```

## Using the RAG solver plugin

[`ovos-openai-plugin`](https://github.com/OpenVoiceOS/ovos-openai-plugin) ships an
`OpenAIRAGSolver` that turns the search + chat round-trip into a single persona
solver: it queries a vector store on this server, injects the retrieved context
into the prompt, and calls the server's `/chat/completions`.

```python
from ovos_solver_openai_persona.rag import OpenAIRAGSolver

rag = OpenAIRAGSolver({
    "api_url": "http://localhost:8337/openai/v1",
    "vector_store_id": store.id,        # from the snippet above
    "llm_model": "my-persona",
    "key": "unused",
    "max_num_results": 3,
})
answer = rag.continue_chat([{"role": "user", "content": "what sits on the mat?"}],
                           lang="en-us")
print(answer)
```

See [`examples/`](../examples/) for runnable versions of both flows and a ready
persona config.

## Endpoint reference

### Files — `/openai/v1/files`

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/openai/v1/files/` | Upload a file (multipart) |
| GET | `/openai/v1/files/` | List files |
| GET | `/openai/v1/files/{file_id}` | Retrieve file metadata |
| GET | `/openai/v1/files/{file_id}/content` | Download file content |
| DELETE | `/openai/v1/files/{file_id}` | Delete a file |

### Vector stores — `/openai/v1/vector_stores`

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/openai/v1/vector_stores/` | Create a vector store |
| GET | `/openai/v1/vector_stores/` | List vector stores |
| GET | `/openai/v1/vector_stores/{id}` | Retrieve a vector store |
| POST | `/openai/v1/vector_stores/{id}` | Modify (name / metadata / expiry) |
| DELETE | `/openai/v1/vector_stores/{id}` | Delete a vector store |
| POST | `/openai/v1/vector_stores/{id}/files` | Attach a file (chunk + embed) |
| GET | `/openai/v1/vector_stores/{id}/files` | List attached files |
| POST | `/openai/v1/vector_stores/{id}/search` | Similarity search |

Legacy `/v1/...` aliases remain (with a `Deprecation` header) — see
[deprecation.md](deprecation.md).

## Notes & current limits

- Each vector store is a collection in the configured `EmbeddingsDB`; deleting the
  store drops the collection.
- Chunking and reranking come from `ovos-document-chunkers` and an OVOS reranker
  plugin (the `[rag]` extra).
- Image embeddings are not yet wired (text only); `encoding_format="base64"` on the
  OpenAI embeddings endpoint is not yet implemented.
- The vector DB plugin must expose the collection API (`create_collection`,
  collection-scoped `query`); `ovos-chromadb-embeddings-plugin` ≥ 0.3.x and
  `ovos-qdrant-embeddings-plugin` are tested.
