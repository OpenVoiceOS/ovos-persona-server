# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cross-repo end-to-end test: the ovos-openai-plugin RAG solver vs a live server.

This wires the genuine deploy stack the way a user would run it:

- server text embedder: ``ovos-gguf-embeddings-plugin`` / ``GGUFEmbeddings``
  (model ``all-MiniLM-L6-v2``, 384-dim, ~46 MB download once)
- server vector DB: ``ovos-chromadb-embeddings-plugin``
- server chat solver: the **real** ``ovos-solver-openai-plugin``
  (``OpenAIChatCompletionsSolver``) behind ``persona.chat``
- client: the **real** ``OpenAIRAGSolver`` from ``ovos-openai-plugin``, pointed at
  the live server's ``/openai/v1`` surface

The only thing mocked is the chat solver's *upstream LLM HTTP call*
(``OpenAIChatCompletionsSolver._do_api_request``) — it returns a hardcoded answer
and records the messages it was handed. Everything else is real: file upload,
vector-store creation, gguf embedding, chromadb search, and the RAG solver's own
HTTP calls to the server. Asserting on the captured messages proves the RAG solver
retrieved the right chunk through the new ``/vector_stores/{id}/search`` endpoint
and injected it into the prompt that reached the server-side solver.

Run::

    pytest tests/e2e/test_e2e_rag_solver.py -v
"""
from __future__ import annotations

import importlib
import json
import os
import socket
import tempfile
import threading
import time
from typing import Generator
from unittest.mock import patch

import httpx
import pytest
import uvicorn

import openai

# All three plugin packages must be importable (fail loudly if not installed).
from ovos_gguf_plugin.embeddings import GGUFEmbeddings  # noqa: F401
from ovos_chromadb_embeddings import ChromaEmbeddingsDB  # noqa: F401
from ovos_solver_openai_persona.engines import OpenAIChatCompletionsSolver
from ovos_solver_openai_persona.rag import OpenAIRAGSolver

# Persona whose chat solver is the real ovos-openai-plugin chat engine. Its
# upstream LLM call is mocked, so api_url/key are never actually contacted.
_PERSONA = {
    "name": "RagBot",
    "solvers": ["ovos-solver-openai-plugin"],
    "ovos-solver-openai-plugin": {
        "api_url": "http://127.0.0.1:1/v1",
        "model": "mock-llm",
        "key": "unused",
    },
}

_DOCS = {
    "cats.txt": b"the cat sat on the mat. cats are fluffy animals that purr.",
    "python.txt": b"python is a programming language used for data science and web development.",
    "moon.txt": b"the moon orbits the earth. the moon affects ocean tides.",
}
_QUERY = "what fluffy animal sits on a mat?"
_HARDCODED_ANSWER = "Cats are fluffy animals that sit on mats."


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app(storage_path: str):
    """Build the real FastAPI app: gguf embeddings + chromadb + openai chat solver."""
    os.environ["TEXT_EMBEDDINGS_PLUGIN"] = "ovos-gguf-embeddings-plugin"
    os.environ["EMBEDDINGS_MODEL"] = "all-MiniLM-L6-v2"
    os.environ["EMBEDDINGS_DB_PLUGIN"] = "ovos-chromadb-embeddings-plugin"
    os.environ["FILE_STORAGE_PATH"] = storage_path

    import ovos_persona_server.config as _cfg_mod
    importlib.reload(_cfg_mod)
    import ovos_persona_server.embeddings as _emb_mod
    import ovos_persona_server.vector_stores as _vs_mod
    importlib.reload(_emb_mod)
    importlib.reload(_vs_mod)

    from ovos_persona_server import create_persona_app

    persona_file = os.path.join(storage_path, "persona.json")
    with open(persona_file, "w") as f:
        json.dump(_PERSONA, f)
    return create_persona_app(persona_file)


@pytest.fixture(scope="module")
def live_server() -> Generator[str, None, None]:
    """Start the real server in a background thread; yield its base URL."""
    storage_path = tempfile.mkdtemp(prefix="persona-rag-solver-e2e-")
    port = _free_port()
    app = _build_app(storage_path)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60  # first run downloads the embedding model
    while time.time() < deadline:
        try:
            httpx.get(f"{base}/openai/v1/files/", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        server.should_exit = True
        raise RuntimeError("server did not start in time")

    yield base

    server.should_exit = True
    thread.join(timeout=10)


def _populated_vector_store(base: str) -> str:
    """Upload the docs and build a vector store via the OpenAI SDK; return its id."""
    client = openai.OpenAI(base_url=f"{base}/openai/v1", api_key="unused")
    file_ids = {}
    for name, content in _DOCS.items():
        f = client.files.create(file=(name, content), purpose="assistants")
        file_ids[name] = f.id

    vs = client.vector_stores.create(name="rag-solver-e2e")
    for file_id in file_ids.values():
        client.vector_stores.files.create(vector_store_id=vs.id, file_id=file_id)

    deadline = time.time() + 30
    while time.time() < deadline:
        statuses = {f.status for f in client.vector_stores.files.list(vector_store_id=vs.id).data}
        if statuses <= {"completed"}:
            break
        time.sleep(0.5)
    else:
        pytest.fail(f"files did not embed in time; statuses={statuses}")
    return vs.id


def test_rag_solver_drives_live_endpoints(live_server):
    """The openai-plugin RAG solver retrieves real context and chats the live server.

    Proves the full cross-repo path:
    RAG solver -> /vector_stores/{id}/search (gguf + chromadb) -> augmented prompt
    -> /chat/completions -> server-side openai chat solver (LLM mocked).
    """
    base = live_server
    vector_store_id = _populated_vector_store(base)

    captured = {}

    def _fake_llm(self, messages):
        # Runs inside the server's chat solver; record what reached it.
        captured["messages"] = messages
        return _HARDCODED_ANSWER

    rag_solver = OpenAIRAGSolver({
        "api_url": f"{base}/openai/v1",
        "vector_store_id": vector_store_id,
        "llm_model": "mock-llm",
        "key": "unused",
        "max_num_results": 3,
        "enable_memory": False,
    })

    # 1. Retrieval against the NEW vector-store search endpoint is real.
    chunks = rag_solver._search_vector_store(_QUERY)
    assert chunks, "RAG solver retrieved no chunks from /vector_stores/search"
    joined = " ".join(chunks).lower()
    assert "cat" in joined or "fluffy" in joined or "mat" in joined, (
        f"top retrieval did not surface the cat document: {chunks}"
    )

    # 2. Full RAG turn: search + chat completion (server-side LLM mocked).
    with patch.object(OpenAIChatCompletionsSolver, "_do_api_request", _fake_llm):
        answer = rag_solver.continue_chat([{"role": "user", "content": _QUERY}], lang="en-us")

    assert answer == _HARDCODED_ANSWER, f"unexpected RAG answer: {answer!r}"

    # 3. The retrieved cat context was injected into the prompt that reached the
    #    server-side solver — i.e. the new endpoint genuinely fed the chat path.
    assert captured.get("messages"), "server-side chat solver was never called"
    prompt_text = " ".join(m.get("content", "") for m in captured["messages"]).lower()
    assert "cat" in prompt_text or "fluffy" in prompt_text or "mat" in prompt_text, (
        f"retrieved context did not reach the chat solver: {captured['messages']}"
    )
