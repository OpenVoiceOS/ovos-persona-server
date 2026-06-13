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
"""End-to-end tests for the shared embeddings backend, via official SDKs.

Drives the official ``openai`` and ``ollama`` clients against a live
ovos-persona-server built from ``create_persona_app`` (the deployed app
factory), exercising the OpenAI ``/openai/v1/embeddings`` and the Ollama
``/ollama/api/embed`` + ``/ollama/api/embeddings`` endpoints exactly as a real
client would. The embeddings backend is injected as a fake so no embeddings
plugin download is needed — the point is that one shared, swappable service
backs every vendor surface.

Run in isolation::

    pytest tests/e2e/test_e2e_embeddings.py -v
"""
from __future__ import annotations

import os
import socket
import tempfile
import threading
import time

os.environ.setdefault("FILE_STORAGE_PATH", tempfile.mkdtemp(prefix="persona-embed-e2e-"))

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

_PERSONA = {"name": "Failer", "solvers": ["ovos-solver-failure-plugin"]}


class _FakeEmbedder:
    config = {"model": "fake-embed"}

    def get_embeddings(self, text):
        return [float(len(text)), 1.0, 2.0]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app() -> FastAPI:
    import json

    from ovos_persona_server import create_persona_app
    from ovos_persona_server.embeddings import get_embeddings_backend

    persona_file = os.path.join(os.environ["FILE_STORAGE_PATH"], "persona.json")
    with open(persona_file, "w") as f:
        json.dump(_PERSONA, f)

    app = create_persona_app(persona_file)
    # one shared, swappable backend behind every vendor router — inject a fake
    # so the test needs no embeddings-plugin download.
    app.dependency_overrides[get_embeddings_backend] = lambda: _FakeEmbedder()
    return app


@pytest.fixture(scope="module")
def base_url():
    port = _free_port()
    config = uvicorn.Config(_build_app(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 15
    base = f"http://127.0.0.1:{port}"
    while time.time() < deadline:
        try:
            httpx.get(f"{base}/openai/v1/models", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        raise RuntimeError("server did not start in time")

    yield base

    server.should_exit = True
    thread.join(timeout=5)


def test_openai_sdk_embeddings(base_url):
    openai = pytest.importorskip("openai", reason="openai SDK not installed")

    client = openai.OpenAI(base_url=f"{base_url}/openai/v1", api_key="not-needed")
    resp = client.embeddings.create(model="text-embedding-ada-002", input=["hello", "world"])
    assert len(resp.data) == 2
    assert resp.data[0].embedding == [5.0, 1.0, 2.0]
    assert resp.data[1].index == 1


def test_ollama_sdk_embed(base_url):
    ollama = pytest.importorskip("ollama", reason="ollama SDK not installed")

    client = ollama.Client(host=f"{base_url}/ollama")
    resp = client.embed(model="nomic-embed-text", input=["a", "bb"])
    assert list(resp.embeddings) == [[1.0, 1.0, 2.0], [2.0, 1.0, 2.0]]


def test_ollama_sdk_legacy_embeddings(base_url):
    ollama = pytest.importorskip("ollama", reason="ollama SDK not installed")

    client = ollama.Client(host=f"{base_url}/ollama")
    resp = client.embeddings(model="nomic-embed-text", prompt="hello")
    assert list(resp.embedding) == [5.0, 1.0, 2.0]
