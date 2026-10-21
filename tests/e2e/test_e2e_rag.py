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
"""End-to-end tests for the OpenAI-compatible Files + Vector Stores surface.

Drives the official ``openai`` SDK against a live ovos-persona-server built
from ``create_persona_app`` (the deployed app factory), exercising the
``/openai/v1/files`` and ``/openai/v1/vector_stores`` endpoints exactly as a
real client would. Files hit the real SQLite-backed store; the vector-store
backend is injected as a fake so no embeddings-DB plugin download is needed.

Run in isolation::

    pytest tests/e2e/test_e2e_rag.py -v
"""
from __future__ import annotations

import os
import socket
import tempfile
import threading
import time

# Isolate the SQLite store + uploads to a temp dir before the app imports the
# metadata module (which builds the engine at import time).
os.environ.setdefault("FILE_STORAGE_PATH", tempfile.mkdtemp(prefix="persona-rag-e2e-"))

from unittest.mock import MagicMock

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

import openai

_PERSONA = {"name": "Failer", "solvers": ["ovos-solver-failure-plugin"]}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app() -> FastAPI:
    import json

    from ovos_persona_server import create_persona_app
    from ovos_persona_server.vector_stores import get_vector_db

    persona_file = os.path.join(os.environ["FILE_STORAGE_PATH"], "persona.json")
    with open(persona_file, "w") as f:
        json.dump(_PERSONA, f)

    app = create_persona_app(persona_file)
    # the vector-store metadata lifecycle (create/list/retrieve/delete) is
    # SQLite-backed; inject a stand-in embeddings DB so no plugin is required.
    app.dependency_overrides[get_vector_db] = lambda: MagicMock()
    return app


@pytest.fixture(scope="module")
def client():
    port = _free_port()
    config = uvicorn.Config(_build_app(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 15
    base = f"http://127.0.0.1:{port}"
    while time.time() < deadline:
        try:
            # any served route confirms startup (and the lifespan db init ran)
            httpx.get(f"{base}/openai/v1/files/", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        raise RuntimeError("server did not start in time")

    yield openai.OpenAI(base_url=f"{base}/openai/v1", api_key="not-needed")

    server.should_exit = True
    thread.join(timeout=5)


def test_files_upload_list_retrieve_delete(client):
    created = client.files.create(file=("notes.txt", b"hello from persona"), purpose="assistants")
    assert created.id
    assert created.purpose == "assistants"

    listed = client.files.list()
    assert any(f.id == created.id for f in listed.data)

    fetched = client.files.retrieve(created.id)
    assert fetched.id == created.id

    deleted = client.files.delete(created.id)
    assert deleted.deleted is True


def test_vector_store_create_list_retrieve_delete(client):
    vs = client.vector_stores.create(name="e2e-store")
    assert vs.id
    assert vs.name == "e2e-store"

    listed = client.vector_stores.list()
    assert any(s.id == vs.id for s in listed.data)

    fetched = client.vector_stores.retrieve(vs.id)
    assert fetched.id == vs.id

    deleted = client.vector_stores.delete(vs.id)
    assert deleted.deleted is True
