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
"""End-to-end tests for the canonical OpenAI-compatible surface.

Drives the official ``openai`` SDK against a live ovos-persona-server built
from ``chat_router`` with a mocked Persona, served by uvicorn on a free port.
Covers the chat completions, completions and models endpoints that ship on
dev under ``/openai/v1`` (previously covered only by mocked unit tests).

Run in isolation::

    pytest tests/e2e/test_e2e_openai.py -v
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

import openai

_REPLY = "The capital of France is Paris."
_CHUNKS = ["The ", "capital ", "of ", "France ", "is ", "Paris."]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app() -> FastAPI:
    from ovos_persona_server.chat import chat_router
    import ovos_persona_server.persona as persona_mod
    from ovos_persona_server.persona import get_default_persona

    # These tests mount the router standalone and supply the persona via
    # dependency_overrides, so they rely on available_personas() falling
    # back to the injected persona. That fallback only applies when the
    # process-wide registry is EMPTY, and any earlier module that built a
    # real app (tests/e2e/test_e2e_embeddings.py) has already populated it
    # -- the registry is process-global and outlives the module. Clear it
    # so this module's expectations hold regardless of test order.
    persona_mod.personas.clear()

    persona = MagicMock()
    persona.name = "test-persona"
    persona.chat.return_value = _REPLY
    persona.stream.side_effect = lambda messages, **kwargs: iter(_CHUNKS)

    # the router lifespan reads the module-global persona at startup
    persona_mod.default_persona = persona

    app = FastAPI()
    app.include_router(chat_router)
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
            httpx.get(f"{base}/openai/v1/models", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        raise RuntimeError("server did not start in time")

    yield openai.OpenAI(base_url=f"{base}/openai/v1", api_key="not-needed")

    server.should_exit = True
    thread.join(timeout=5)


def test_chat_completion(client):
    resp = client.chat.completions.create(
        model="test-persona",
        messages=[{"role": "user", "content": "capital of France?"}],
    )
    assert resp.choices[0].message.content == _REPLY


def test_chat_completion_streaming(client):
    chunks = []
    for ev in client.chat.completions.create(
        model="test-persona",
        messages=[{"role": "user", "content": "capital of France?"}],
        stream=True,
    ):
        delta = ev.choices[0].delta.content
        if delta:
            chunks.append(delta)
    assert "".join(chunks) == "".join(_CHUNKS)


def test_completions(client):
    resp = client.completions.create(model="test-persona", prompt="capital of France?")
    assert resp.choices[0].text == _REPLY


def test_models_list(client):
    models = client.models.list()
    assert any(m.id == "test-persona" for m in models.data)
