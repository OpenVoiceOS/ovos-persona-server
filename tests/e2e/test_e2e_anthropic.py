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
"""End-to-end tests for the Anthropic-compatible API surface.

These exercise the real deployment path a user hits in production: the
official ``anthropic`` Python SDK pointed at a live ovos-persona-server
instance. A minimal FastAPI app is built from ``anthropic_router`` with a
mocked Persona, served by uvicorn on a free port in a background thread, and
driven by ``anthropic.Anthropic(base_url=...)`` exactly as a real client would.

Run in isolation::

    pytest tests/e2e/test_e2e_anthropic.py -v
"""
from __future__ import annotations

import importlib.util
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from unittest.mock import MagicMock

import anthropic

_CHAT_REPLY = "The capital of France is Paris."
_STREAM_CHUNKS = ["The ", "capital ", "of ", "France ", "is ", "Paris."]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app() -> FastAPI:
    from ovos_persona_server.anthropic import anthropic_router
    from ovos_persona_server.persona import get_default_persona

    persona = MagicMock()
    persona.name = "test-persona"
    persona.chat.return_value = _CHAT_REPLY
    # fresh iterator per call so multiple streaming requests work
    persona.stream.side_effect = lambda messages, **kwargs: iter(_STREAM_CHUNKS)

    app = FastAPI()
    app.include_router(anthropic_router)

    @app.get("/health")
    def _health() -> dict:
        return {"ok": True}

    app.dependency_overrides[get_default_persona] = lambda: persona
    return app


@pytest.fixture(scope="module")
def client() -> anthropic.Anthropic:
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

    # the SDK appends /v1/messages; our router prefix is /anthropic/v1
    yield anthropic.Anthropic(base_url=f"{base}/anthropic", api_key="test-key")

    server.should_exit = True
    thread.join(timeout=5)


def test_messages_create_roundtrip(client):
    msg = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=128,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    assert msg.role == "assistant"
    assert msg.content[0].type == "text"
    assert msg.content[0].text == _CHAT_REPLY
    assert msg.model == "claude-3-opus-20240229"
    assert msg.usage.output_tokens > 0


def test_messages_with_system_prompt(client):
    msg = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=64,
        system="You are a geography teacher.",
        messages=[{"role": "user", "content": "capital of France?"}],
    )
    assert msg.content[0].text == _CHAT_REPLY


def test_messages_multi_turn(client):
    msg = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=64,
        messages=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "capital of France?"},
        ],
    )
    assert msg.content[0].text == _CHAT_REPLY


def test_messages_streaming(client):
    chunks = []
    with client.messages.stream(
        model="claude-3-opus-20240229",
        max_tokens=128,
        messages=[{"role": "user", "content": "capital of France?"}],
    ) as stream:
        for text in stream.text_stream:
            chunks.append(text)
        final = stream.get_final_message()
    assert "".join(chunks) == "".join(_STREAM_CHUNKS)
    assert final.role == "assistant"
