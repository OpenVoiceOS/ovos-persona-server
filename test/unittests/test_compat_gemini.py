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
    from ovos_persona_server.gemini import gemini_router
    from ovos_persona_server.deprecated_routers import (
        add_deprecation_middleware,
        register_deprecated_routes,
    )

    app = FastAPI()
    app.include_router(chat_router)         # /openai/v1/...
    app.include_router(ollama_router)       # /ollama/api/...
    app.include_router(gemini_router)       # /gemini/v1beta/models/...
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

# Gemini  (prefix: /gemini/v1beta/models)
# ---------------------------------------------------------------------------

class TestGeminiRouter:
    def test_generate_content(self, client):
        resp = client.post(
            "/gemini/v1beta/models/gemini-pro:generateContent?key=fake",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["candidates"][0]["content"]["parts"][0]["text"] == "hello from fake persona"

    def test_stream_generate_content(self, client):
        resp = client.post(
            "/gemini/v1beta/models/gemini-pro:streamGenerateContent?key=fake",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_empty_contents_rejected(self, client):
        resp = client.post(
            "/gemini/v1beta/models/gemini-pro:generateContent",
            json={"contents": []},
        )
        assert resp.status_code == 422

    def test_model_role_accepted(self, client):
        resp = client.post(
            "/gemini/v1beta/models/gemini-pro:generateContent",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": "hi"}]},
                    {"role": "model", "parts": [{"text": "hello"}]},
                    {"role": "user", "parts": [{"text": "how are you"}]},
                ]
            },
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Cohere  (prefix: /cohere/v1)
# ---------------------------------------------------------------------------

