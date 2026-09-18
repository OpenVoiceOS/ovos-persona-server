# Examples

Runnable examples for `ovos-persona-server`, focused on the RAG surface
(files + vector stores + embeddings). See [`docs/rag.md`](../docs/rag.md) for the
full reference.

## Start a RAG-capable server

```bash
uv pip install ovos-persona-server[rag] \
  ovos-gguf-plugin ovos-chromadb-embeddings-plugin

# gguf embeddings + chromadb vector store, OpenAI chat solver persona
TEXT_EMBEDDINGS_PLUGIN=ovos-gguf-embeddings-plugin \
EMBEDDINGS_MODEL=all-MiniLM-L6-v2 \
EMBEDDINGS_DB_PLUGIN=ovos-chromadb-embeddings-plugin \
PERSONA_PATH=examples/persona_rag.json \
python -m ovos_persona_server
```

| File | What it shows |
| --- | --- |
| [`persona_rag.json`](persona_rag.json) | A persona whose chat backend is an OpenAI-compatible LLM |
| [`rag_files_vector_stores.py`](rag_files_vector_stores.py) | Upload → vector store → search with the `openai` SDK |
| [`rag_solver_plugin.py`](rag_solver_plugin.py) | Drive the whole RAG turn via `OpenAIRAGSolver` |
| [`embeddings_clients.py`](embeddings_clients.py) | The shared embeddings backend across vendor SDKs |
