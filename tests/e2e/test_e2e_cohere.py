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
"""End-to-end tests for the Cohere-compatible API surface.

Drives the official ``cohere`` Python SDK (``cohere.Client(base_url=...)``,
the v1 client that targets ``/v1/chat`` and ``/v1/generate``) against a live
ovos-persona-server instance built from ``cohere_router`` with a mocked
Persona and served by uvicorn on a free port.

Run in isolation::

    pytest tests/e2e/test_e2e_cohere.py -v
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

cohere = pytest.importorskip("cohere", reason="cohere SDK not installed")

_CHAT_REPLY = "The capital of France is Paris."
_STREAM_CHUNKS = ["The ", "capital ", "of ", "France ", "is ", "Paris."]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app() -> FastAPI:
    from ovos_persona_server.cohere import cohere_router
    from ovos_persona_server.persona import get_default_persona

    persona = MagicMock()
    persona.name = "test-persona"
    persona.chat.return_value = _CHAT_REPLY
    persona.stream.side_effect = lambda messages, **kwargs: iter(_STREAM_CHUNKS)

    app = FastAPI()
    app.include_router(cohere_router)

    @app.get("/health")
    def _health() -> dict:
        return {"ok": True}

    app.dependency_overrides[get_default_persona] = lambda: persona
    return app


@pytest.fixture(scope="module")
def client():
    port = _free_port()
    config = uvicorn.Config(_build_app(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    base = f"http://127.0.0.1:{port}"
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        raise RuntimeError("server did not start in time")

    # the v1 SDK appends /v1/chat, /v1/generate; our router prefix is /cohere/v1
    yield cohere.Client(api_key="test-key", base_url=f"{base}/cohere")

    server.should_exit = True
    thread.join(timeout=5)


def test_chat_roundtrip(client):
    resp = client.chat(model="command-r", message="What is the capital of France?")
    assert resp.text == _CHAT_REPLY


def test_chat_with_preamble_and_history(client):
    resp = client.chat(
        model="command-r",
        preamble="You are a geography teacher.",
        chat_history=[
            {"role": "USER", "message": "Hi"},
            {"role": "CHATBOT", "message": "Hello!"},
        ],
        message="capital of France?",
    )
    assert resp.text == _CHAT_REPLY


def test_generate_roundtrip(client):
    resp = client.generate(model="command", prompt="capital of France?")
    assert resp.generations[0].text == _CHAT_REPLY


def test_chat_stream(client):
    chunks = []
    for event in client.chat_stream(model="command-r", message="capital of France?"):
        if event.event_type == "text-generation":
            chunks.append(event.text)
    assert "".join(chunks) == "".join(_STREAM_CHUNKS)
