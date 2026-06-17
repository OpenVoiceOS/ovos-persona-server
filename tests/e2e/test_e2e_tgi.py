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
"""End-to-end tests for the HuggingFace TGI-compatible API surface.

Drives the official maintained client ``huggingface_hub.InferenceClient``
(``InferenceClient(model=<endpoint>)``) against a live ovos-persona-server
instance built from ``tgi_router`` with a mocked Persona and served by
uvicorn on a free port. The client posts to the endpoint root and selects
streaming via the ``stream`` flag, exactly as it does against a real TGI
deployment.

Run in isolation::

    pytest tests/e2e/test_e2e_tgi.py -v
"""
from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from unittest.mock import MagicMock

hf = pytest.importorskip("huggingface_hub", reason="huggingface_hub not installed")
from huggingface_hub import InferenceClient  # noqa: E402

_CHAT_REPLY = "The capital of France is Paris."
_STREAM_CHUNKS = ["The ", "capital ", "of ", "France ", "is ", "Paris."]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app() -> FastAPI:
    from ovos_persona_server.huggingface_tgi import tgi_router
    from ovos_persona_server.persona import get_default_persona

    persona = MagicMock()
    persona.name = "test-persona"
    persona.chat.return_value = _CHAT_REPLY
    persona.stream.side_effect = lambda messages: iter(_STREAM_CHUNKS)

    app = FastAPI()
    app.include_router(tgi_router)
    app.dependency_overrides[get_default_persona] = lambda: persona
    return app


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    config = uvicorn.Config(_build_app(), host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    base = f"http://127.0.0.1:{port}"
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/tgi/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        srv.should_exit = True
        raise RuntimeError("server did not start in time")

    yield base

    srv.should_exit = True
    thread.join(timeout=5)


def test_text_generation_roundtrip(server):
    client = InferenceClient(model=f"{server}/tgi")
    out = client.text_generation("What is the capital of France?", max_new_tokens=32)
    assert out == _CHAT_REPLY


def test_text_generation_with_details(server):
    client = InferenceClient(model=f"{server}/tgi")
    resp = client.text_generation("capital of France?", max_new_tokens=32, details=True)
    assert resp.generated_text == _CHAT_REPLY
    assert resp.details.finish_reason


def test_text_generation_stream(server):
    client = InferenceClient(model=f"{server}/tgi")
    tokens = [
        t for t in client.text_generation("capital of France?", max_new_tokens=32, stream=True)
    ]
    assert "".join(tokens) == "".join(_STREAM_CHUNKS)


def test_health_endpoint(server):
    assert httpx.get(f"{server}/tgi/health", timeout=5).status_code == 200


def test_info_endpoint(server):
    info = httpx.get(f"{server}/tgi/info", timeout=5).json()
    assert info["model_id"] == "test-persona"
    assert info["model_pipeline_tag"] == "text-generation"


def test_explicit_generate_route(server):
    """The native TGI /generate route still returns a single object."""
    resp = httpx.post(
        f"{server}/tgi/generate",
        json={"inputs": "capital of France?", "parameters": {"max_new_tokens": 32}},
        timeout=5,
    ).json()
    assert resp["generated_text"] == _CHAT_REPLY
