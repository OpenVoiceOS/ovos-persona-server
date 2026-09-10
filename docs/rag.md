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

## Drop-in replacement for OpenAI

Because the Files, Vector Stores, and Embeddings endpoints match the OpenAI API,
**any third-party app already built on OpenAI's RAG endpoints can point at this server
instead** — just change the `base_url` (and ignore the key). No code changes:

```python
# before: OpenAI cloud
client = OpenAI()                                             # api.openai.com
# after: self-hosted, private, no OpenAI key, swappable local models
client = OpenAI(base_url="http://your-host:8337/openai/v1", api_key="unused")
```

That turns an OpenAI-dependent RAG app into a self-hosted one backed by OVOS plugins
(local gguf embeddings, chromadb/qdrant) — private data never leaves your box, no
per-token cost, and you choose the embedding/vector-DB providers. The same applies to
the chat/embeddings surfaces, so an app using OpenAI for chat **and** RAG can be
migrated wholesale by switching one URL.

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

## Using RAG as a persona memory plugin

[`ovos-openai-plugin`](https://github.com/OpenVoiceOS/ovos-openai-plugin) ships
`PersonaServerRAGMemory` (`ovos-openai-rag-memory-plugin`) — an OVOS persona
**memory plugin** (`AgentContextManager`). Instead of owning the chat round-trip, it
hooks the persona's context-building step: it searches a vector store on this server
and injects the retrieved chunks into the conversation, then the persona's normal
chat engine answers. This composes with any chat backend.

Set it as the persona's `memory_module` (the persona passes the config block to the
plugin):

```json
{
  "name": "kb-assistant",
  "solvers": ["ovos-solver-openai-plugin"],
  "memory_module": "ovos-openai-rag-memory-plugin",
  "ovos-openai-rag-memory-plugin": {
    "api_url": "http://localhost:8337/openai/v1",
    "vector_store_id": "vs_...",
    "inject_mode": "system",
    "retrieval": {"max_num_results": 5}
  }
}
```

`inject_mode` selects how retrieved context enters the prompt (`system` — a separate
system message, default; `system_prompt`; `developer`; or `user`), plus configurable
retrieval and formatting — see the plugin's module docstring. Requires `ovos-persona`
with memory-plugin config passing.

See [`examples/`](../examples/) for runnable flows and a ready persona config.

## Backend vs hosted agent — the `CHAT_MEMORY` toggle

Whether the chat endpoints apply that `memory_module` (RAG or short/long-term
memory) on the way in is a **server-side deployment choice**, not a per-request one.
It is governed by the `CHAT_MEMORY` environment variable:

| `CHAT_MEMORY` | Mode | Behaviour |
| --- | --- | --- |
| `off` *(default)* | **Backend** | Stateless passthrough. The client sends the full message list and owns conversation state; the server applies **no** memory. The client is expected to drive the Files / Vector-Stores endpoints itself (that is what exposing them is *for*). |
| `transparent` | **Hosted agent** | The server owns state: it treats the latest user message as the new turn, folds the persona's `memory_module` (history + RAG) into every request keyed by session, and persists the exchange for the next call. |

```bash
# backend (multi-user / drop-in OpenAI replacement) — the default
ovos-persona-server --persona kb.json

# single-user hosted agent — transparent server-side memory/RAG
CHAT_MEMORY=transparent ovos-persona-server --persona kb.json
```

**Why off by default.** As a *drop-in OpenAI replacement* the server must behave like
a backend: a shared server-side memory would leak conversation across callers in a
multi-user deployment, and the OpenAI contract expects the client to send history. So
`off` is correct unless you are deploying a single-user agent.

**Per-user namespacing.** In `transparent` mode the OpenAI `user` field (when present)
is used as the memory session key, so distinct callers keep separate histories; absent
it, a single default session is used. Even so, `transparent` is intended for the
hosted-agent case — keep it `off` for general multi-tenant backends.

The legacy `/v1/completions` surface reads the same `user` field. On the A2A
surface the key is `contextId`, the conversation identifier the protocol already
carries. A client that echoes the `contextId` it got back keeps its history; one
that sends none is given a fresh identifier per request by the a2a-sdk, so every
request starts a new conversation. That is the anonymous rule again — an
unidentified caller is independent rather than shared — but on A2A it applies per
request rather than per process, so a client that wants continuity has to echo the
id rather than rely on the server remembering it.

> Tool/function-calling requests (`tools=`) are always a stateless passthrough
> regardless of this toggle — the client drives the tool loop and owns that state.

This applies uniformly to every vendor chat surface (OpenAI, Ollama, Cohere, Gemini,
Anthropic, …), since they all route through the same persona chat path.

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
