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
"""Real cross-plugin end-to-end RAG test.

Drives ``create_persona_app`` with the genuine OVOS embedding stack:
- text embedder: ``ovos-gguf-embeddings-plugin`` / ``GGUFEmbeddings``
  (model: ``all-MiniLM-L6-v2``, 384-dim, ~46 MB download once)
- vector DB: ``ovos-chromadb-embeddings-plugin`` and
  ``ovos-qdrant-embeddings-plugin`` (parametrised)

No dependency overrides – every component is real.

Run::

    pytest tests/e2e/test_e2e_rag_plugins.py -v
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
from typing import Generator

import httpx
import pytest
import uvicorn

openai = pytest.importorskip("openai", reason="openai SDK not installed")
pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed")

# ---------------------------------------------------------------------------
# Make sure the three plugin packages are importable (fail loudly if not)
# ---------------------------------------------------------------------------
from ovos_gguf_plugin.embeddings import GGUFEmbeddings  # noqa: F401 – must import
from ovos_chromadb_embeddings import ChromaEmbeddingsDB  # noqa: F401
from ovos_qdrant_embeddings import QdrantEmbeddingsDB  # noqa: F401

_PERSONA = {"name": "Failer", "solvers": ["ovos-solver-failure-plugin"]}

_DB_PARAMS = [
    "ovos-chromadb-embeddings-plugin",
    "ovos-qdrant-embeddings-plugin",
]

# Documents with clearly distinct topics
_DOCS = {
    "cats.txt": b"the cat sat on the mat. cats are fluffy animals that purr.",
    "python.txt": b"python is a programming language. python is used for data science and web development.",
    "moon.txt": b"the moon orbits the earth. the moon affects ocean tides.",
}
_QUERY_CAT = "fluffy animal sitting on a mat"
_EXPECTED_FILENAME = "cats.txt"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app(db_plugin: str, storage_path: str):
    """Build the real FastAPI app with the given DB plugin and storage path."""
    from fastapi import FastAPI

    # Environment must be set BEFORE any persona-server module is imported
    # inside this process; we patch os.environ and reload the settings singleton.
    os.environ["TEXT_EMBEDDINGS_PLUGIN"] = "ovos-gguf-embeddings-plugin"
    os.environ["EMBEDDINGS_MODEL"] = "all-MiniLM-L6-v2"
    os.environ["EMBEDDINGS_DB_PLUGIN"] = db_plugin
    os.environ["FILE_STORAGE_PATH"] = storage_path

    # Force the settings singleton to re-read the environment
    import importlib
    import ovos_persona_server.config as _cfg_mod
    importlib.reload(_cfg_mod)

    # Also reload modules that captured `settings` at import time
    import ovos_persona_server.embeddings as _emb_mod
    import ovos_persona_server.vector_stores as _vs_mod
    importlib.reload(_emb_mod)
    importlib.reload(_vs_mod)

    from ovos_persona_server import create_persona_app

    persona_file = os.path.join(storage_path, "persona.json")
    with open(persona_file, "w") as f:
        json.dump(_PERSONA, f)

    app = create_persona_app(persona_file)
    return app


@pytest.fixture(
    params=_DB_PARAMS,
    ids=_DB_PARAMS,
    scope="module",
)
def rag_client(request) -> Generator:
    """Parametrised fixture: one server per DB plugin, isolated storage."""
    db_plugin: str = request.param
    storage_path = tempfile.mkdtemp(prefix=f"persona-rag-e2e-{db_plugin.split('-')[1]}-")

    port = _free_port()
    app = _build_app(db_plugin, storage_path)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60  # model download may take a moment on first run
    while time.time() < deadline:
        try:
            httpx.get(f"{base}/openai/v1/files/", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        server.should_exit = True
        raise RuntimeError(f"server ({db_plugin}) did not start in time")

    client = openai.OpenAI(base_url=f"{base}/openai/v1", api_key="not-needed")
    yield client, db_plugin

    server.should_exit = True
    thread.join(timeout=10)


def test_rag_search_ranks_correct_doc(rag_client):
    """Upload 3 distinct docs, search with a cat-related query, expect cats.txt first."""
    client, db_plugin = rag_client

    # 1. Upload files
    file_ids: dict[str, str] = {}
    for filename, content in _DOCS.items():
        f = client.files.create(file=(filename, content), purpose="assistants")
        assert f.id, f"[{db_plugin}] file upload returned no id"
        file_ids[filename] = f.id

    # 2. Create a vector store
    vs = client.vector_stores.create(name=f"e2e-store-{db_plugin}")
    assert vs.id, f"[{db_plugin}] vector store creation returned no id"

    # 3. Add all files to the vector store
    for filename, file_id in file_ids.items():
        vsf = client.vector_stores.files.create(
            vector_store_id=vs.id,
            file_id=file_id,
        )
        assert vsf.status in ("completed", "in_progress"), (
            f"[{db_plugin}] unexpected vsf status for {filename}: {vsf.status}"
        )

    # 4. Poll until all files are completed (embedding is synchronous but let's be safe)
    deadline = time.time() + 30
    while time.time() < deadline:
        files_list = client.vector_stores.files.list(vector_store_id=vs.id)
        statuses = {f.status for f in files_list.data}
        if statuses <= {"completed"}:
            break
        time.sleep(0.5)
    else:
        pytest.fail(f"[{db_plugin}] files did not reach 'completed' in time; statuses={statuses}")

    # 5. Search
    results = client.vector_stores.search(
        vector_store_id=vs.id,
        query=_QUERY_CAT,
        max_num_results=3,
    )

    assert results.data, f"[{db_plugin}] search returned no results"

    # Retrieve the file_id of the top result and map back to filename
    top_file_id = results.data[0].file_id
    id_to_name = {v: k for k, v in file_ids.items()}
    top_filename = id_to_name.get(top_file_id, "<unknown>")

    assert top_filename == _EXPECTED_FILENAME, (
        f"[{db_plugin}] expected '{_EXPECTED_FILENAME}' at rank-1, "
        f"got '{top_filename}' (score={results.data[0].score:.4f}). "
        f"All results: {[(id_to_name.get(r.file_id, r.file_id), r.score) for r in results.data]}"
    )
