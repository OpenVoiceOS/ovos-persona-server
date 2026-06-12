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
"""End-to-end tests for the Google Gemini-compatible API surface.

Drives the official ``google-genai`` SDK
(``genai.Client(http_options=HttpOptions(base_url=...))``) against a live
ovos-persona-server instance built from ``gemini_router`` with a mocked
Persona and served by uvicorn on a free port.

Run in isolation::

    pytest tests/e2e/test_e2e_gemini.py -v
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

genai = pytest.importorskip("google.genai", reason="google-genai SDK not installed")
from google.genai import types  # noqa: E402

_CHAT_REPLY = "The capital of France is Paris."
_STREAM_CHUNKS = ["The ", "capital ", "of ", "France ", "is ", "Paris."]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app() -> FastAPI:
    from ovos_persona_server.gemini import gemini_router
    from ovos_persona_server.persona import get_default_persona

    persona = MagicMock()
    persona.name = "test-persona"
    persona.chat.return_value = _CHAT_REPLY
    persona.stream.side_effect = lambda messages: iter(_STREAM_CHUNKS)

    app = FastAPI()
    app.include_router(gemini_router)

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

    # the SDK appends /v1beta/models/{model}:generateContent; prefix is /gemini
    cli = genai.Client(
        api_key="test-key",
        http_options=types.HttpOptions(base_url=f"{base}/gemini"),
    )
    yield cli

    server.should_exit = True
    thread.join(timeout=5)


def test_generate_content_roundtrip(client):
    resp = client.models.generate_content(
        model="gemini-1.5-flash",
        contents="What is the capital of France?",
    )
    assert resp.text == _CHAT_REPLY
    assert resp.candidates[0].content.role == "model"


def test_generate_content_with_system_instruction(client):
    resp = client.models.generate_content(
        model="gemini-1.5-pro",
        contents="capital of France?",
        config=types.GenerateContentConfig(
            system_instruction="You are a geography teacher.",
        ),
    )
    assert resp.text == _CHAT_REPLY


def test_generate_content_multi_turn(client):
    resp = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text="Hi")]),
            types.Content(role="model", parts=[types.Part(text="Hello!")]),
            types.Content(role="user", parts=[types.Part(text="capital of France?")]),
        ],
    )
    assert resp.text == _CHAT_REPLY


def test_generate_content_stream(client):
    chunks = [
        chunk.text
        for chunk in client.models.generate_content_stream(
            model="gemini-1.5-flash",
            contents="capital of France?",
        )
        if chunk.text
    ]
    assert "".join(chunks) == "".join(_STREAM_CHUNKS)
