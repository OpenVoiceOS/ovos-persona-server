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
"""Unit tests for the shared, swappable embeddings backend across all routers.

The OpenAI (``/openai/v1/embeddings``) and Ollama (``/ollama/api/embed`` and the
legacy ``/ollama/api/embeddings``) endpoints all delegate to the single
``get_embeddings_backend`` dependency. These tests inject a fake backend so no
embeddings plugin needs to be installed, and assert each vendor surface speaks
its own response shape while sharing one backend.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ovos_persona_server.chat import chat_router
from ovos_persona_server.ollama import ollama_router
from ovos_persona_server.embeddings import (
    get_embeddings_backend,
    embed_texts,
    backend_model_name,
    _SolverEmbedder,
    _persona_solver_embedder,
)


class FakeEmbedder:
    """Deterministic stand-in for any ``get_embeddings``-capable backend."""

    config = {"model": "fake-embed"}

    def get_embeddings(self, text):
        # length-derived, fixed-width vector — deterministic and trivially checkable
        return [float(len(text)), 1.0, 2.0]


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(ollama_router)
    app.dependency_overrides[get_embeddings_backend] = lambda: FakeEmbedder()
    return TestClient(app)


# --------------------------------------------------------------------------- #
# OpenAI surface
# --------------------------------------------------------------------------- #

def test_openai_embeddings_list_input(client):
    resp = client.post("/openai/v1/embeddings", json={"input": ["hi", "world"], "model": "x"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["model"] == "x"
    assert len(body["data"]) == 2
    assert body["data"][0]["embedding"] == [2.0, 1.0, 2.0]
    assert body["data"][1]["index"] == 1


def test_openai_embeddings_string_input(client):
    resp = client.post("/openai/v1/embeddings", json={"input": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 1
    assert body["data"][0]["embedding"] == [5.0, 1.0, 2.0]


def test_openai_embeddings_base64_roundtrip(client):
    import base64
    import struct

    resp = client.post("/openai/v1/embeddings",
                       json={"input": "hello", "encoding_format": "base64"})
    assert resp.status_code == 200
    encoded = resp.json()["data"][0]["embedding"]
    assert isinstance(encoded, str)
    raw = base64.b64decode(encoded)
    decoded = list(struct.unpack(f"<{len(raw) // 4}f", raw))
    assert decoded == [5.0, 1.0, 2.0]


# --------------------------------------------------------------------------- #
# Ollama surface
# --------------------------------------------------------------------------- #

def test_ollama_embed_batch(client):
    resp = client.post("/ollama/api/embed", json={"model": "x", "input": ["a", "bb"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "x"
    assert body["embeddings"] == [[1.0, 1.0, 2.0], [2.0, 1.0, 2.0]]


def test_ollama_embed_string(client):
    resp = client.post("/ollama/api/embed", json={"model": "x", "input": "abc"})
    assert resp.status_code == 200
    assert resp.json()["embeddings"] == [[3.0, 1.0, 2.0]]


def test_ollama_legacy_embeddings_prompt(client):
    resp = client.post("/ollama/api/embeddings", json={"model": "x", "prompt": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    # legacy shape: a single "embedding" vector, not a list of vectors
    assert body["embedding"] == [5.0, 1.0, 2.0]
    assert "embeddings" not in body


# --------------------------------------------------------------------------- #
# Backend helpers
# --------------------------------------------------------------------------- #

def test_embed_texts_coerces_to_float():
    class IntEmbedder:
        def get_embeddings(self, text):
            return [1, 2, 3]  # ints, not floats

    out = embed_texts(IntEmbedder(), ["a", "b"])
    assert out == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert all(isinstance(x, float) for vec in out for x in vec)


def test_backend_model_name_prefers_request():
    assert backend_model_name(FakeEmbedder(), "requested") == "requested"
    assert backend_model_name(FakeEmbedder()) == "fake-embed"

    class NoConfig:
        config = {}

    assert backend_model_name(NoConfig()) == "server-default"


def test_solver_embedder_adapts_get_embeddings():
    class Solver:
        def get_embeddings(self, text):
            return [9.0]

    emb = _SolverEmbedder(Solver())
    assert emb.get_embeddings("x") == [9.0]
    assert emb.config == {}


def test_persona_solver_embedder_none_without_persona(monkeypatch):
    import ovos_persona_server.persona as persona_mod
    monkeypatch.setattr(persona_mod, "default_persona", None)
    assert _persona_solver_embedder() is None
