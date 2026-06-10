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
    from ovos_persona_server.cohere import cohere_router
    from ovos_persona_server.deprecated_routers import (
        add_deprecation_middleware,
        register_deprecated_routes,
    )

    app = FastAPI()
    app.include_router(chat_router)         # /openai/v1/...
    app.include_router(ollama_router)       # /ollama/api/...
    app.include_router(cohere_router)       # /cohere/v1/...
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

# Cohere  (prefix: /cohere/v1)
# ---------------------------------------------------------------------------

class TestCohereRouter:
    def test_chat_non_stream(self, client):
        resp = client.post("/cohere/v1/chat", json={"message": "hello"})
        assert resp.status_code == 200
        assert resp.json()["text"] == "hello from fake persona"

    def test_chat_stream(self, client):
        resp = client.post("/cohere/v1/chat", json={"message": "hello", "stream": True})
        assert resp.status_code == 200
        lines = [l.strip() for l in resp.text.splitlines() if l.strip()]
        events = [json.loads(l) for l in lines if l.startswith("{")]
        assert "stream-end" in [e.get("event_type") for e in events]

    def test_chat_empty_message_rejected(self, client):
        resp = client.post("/cohere/v1/chat", json={"message": ""})
        assert resp.status_code == 422

    def test_generate_non_stream(self, client):
        resp = client.post("/cohere/v1/generate", json={"prompt": "once upon a time"})
        assert resp.status_code == 200
        assert "generations" in resp.json()

    def test_temperature_out_of_range_rejected(self, client):
        resp = client.post("/cohere/v1/chat", json={"message": "hi", "temperature": 6.0})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# HuggingFace TGI  (prefix: /tgi)
# ---------------------------------------------------------------------------

