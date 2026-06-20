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
"""End-to-end tests for the canonical Ollama-compatible surface.

Drives the official ``ollama`` Python client against a live ovos-persona-server
built from ``ollama_router`` with a mocked Persona. Covers the chat, generate
and tags endpoints that ship on dev under ``/ollama/api`` (previously covered
only by mocked unit tests).

Run in isolation::

    pytest tests/e2e/test_e2e_ollama.py -v
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

import ollama

_REPLY = "The capital of France is Paris."
_CHUNKS = ["The ", "capital ", "of ", "France ", "is ", "Paris."]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app() -> FastAPI:
    from ovos_persona_server.ollama import ollama_router
    import ovos_persona_server.persona as persona_mod
    from ovos_persona_server.persona import get_default_persona

    persona = MagicMock()
    persona.name = "test-persona"
    persona.chat.return_value = _REPLY
    persona.stream.side_effect = lambda messages, **kwargs: iter(_CHUNKS)

    # the router lifespan reads the module-global persona at startup
    persona_mod.default_persona = persona

    app = FastAPI()
    app.include_router(ollama_router)
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
            httpx.get(f"{base}/ollama/api/tags", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        raise RuntimeError("server did not start in time")

    # the ollama client appends /api/...; our router prefix is /ollama/api
    yield ollama.Client(host=f"{base}/ollama")

    server.should_exit = True
    thread.join(timeout=5)


def test_chat(client):
    resp = client.chat(model="test-persona", messages=[{"role": "user", "content": "capital of France?"}])
    assert resp["message"]["content"] == _REPLY


def test_chat_streaming(client):
    chunks = [
        part["message"]["content"]
        for part in client.chat(
            model="test-persona",
            messages=[{"role": "user", "content": "capital of France?"}],
            stream=True,
        )
        if part["message"]["content"]
    ]
    assert "".join(chunks) == "".join(_CHUNKS)


def test_generate(client):
    resp = client.generate(model="test-persona", prompt="capital of France?")
    assert resp["response"] == _REPLY


def test_list_tags(client):
    listed = client.list()
    names = [m.get("name") or m.get("model") for m in listed["models"]]
    assert any(n and "test-persona" in n for n in names)
