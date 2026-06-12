# Licensed under the Apache License, Version 2.0
"""Unit tests for persona server compatibility routers.

All compat routers mount under a vendor-namespaced prefix to avoid conflicts:
  /v1/...                  OpenAI (chat_router — original prefix)
  /api/...                 Ollama (ollama_router — original prefix)
  /anthropic/v1/...        Anthropic Claude
  /gemini/v1beta/models/.. Google Gemini
  /cohere/v1/...           Cohere
  /tgi/...                 HuggingFace TGI
  /bedrock/model/...       AWS Bedrock

Uses dependency-override to replace the live Persona with a lightweight stub.
"""
import json
from typing import Dict, Generator, List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fake persona
# ---------------------------------------------------------------------------

class FakePersona:
    name: str = "fake-persona"
    config: dict = {}

    def chat(self, messages: List[Dict], **kwargs) -> str:
        return "hello from fake persona"

    def stream(self, messages: List[Dict], **kwargs) -> Generator[str, None, None]:
        for word in ["hello", " ", "streaming"]:
            yield word

    class solvers:
        loaded_modules: dict = {}


FAKE_PERSONA = FakePersona()


def _override_persona():
    return FAKE_PERSONA


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def _make_app() -> FastAPI:
    from ovos_persona_server.persona import get_default_persona
    from ovos_persona_server.chat import chat_router
    from ovos_persona_server.ollama import ollama_router
    from ovos_persona_server.anthropic import anthropic_router
    from ovos_persona_server.deprecated_routers import (
        add_deprecation_middleware,
        register_deprecated_routes,
    )

    app = FastAPI()
    app.include_router(chat_router)         # /openai/v1/...
    app.include_router(ollama_router)       # /ollama/api/...
    app.include_router(anthropic_router)    # /anthropic/v1/...
    register_deprecated_routes(app)         # /v1/... and /api/... (deprecated)
    add_deprecation_middleware(app)
    app.dependency_overrides[get_default_persona] = _override_persona
    return app


@pytest.fixture(scope="module")
def client():
    app = _make_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# OpenAI chat  (canonical prefix: /openai/v1)
# ---------------------------------------------------------------------------

# Anthropic  (prefix: /anthropic/v1)
# ---------------------------------------------------------------------------

class TestAnthropicRouter:
    def test_create_message_non_stream(self, client):
        resp = client.post(
            "/anthropic/v1/messages",
            json={
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
            headers={"x-api-key": "fake-key"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        assert body["content"][0]["text"] == "hello from fake persona"

    def test_create_message_stream(self, client):
        resp = client.post(
            "/anthropic/v1/messages",
            json={
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        events = [l for l in resp.text.splitlines() if l.startswith("event:")]
        event_types = [e.split(": ", 1)[1] for e in events]
        assert "message_start" in event_types
        assert "message_stop" in event_types

    def test_empty_messages_rejected(self, client):
        resp = client.post(
            "/anthropic/v1/messages",
            json={"model": "claude-3-opus-20240229", "messages": [], "max_tokens": 100},
        )
        assert resp.status_code == 422

    def test_zero_max_tokens_rejected(self, client):
        resp = client.post(
            "/anthropic/v1/messages",
            json={
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 0,
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Gemini  (prefix: /gemini/v1beta/models)
# ---------------------------------------------------------------------------

