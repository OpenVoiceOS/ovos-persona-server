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
    from ovos_persona_server.aws_bedrock import bedrock_router
    from ovos_persona_server.deprecated_routers import (
        add_deprecation_middleware,
        register_deprecated_routes,
    )

    app = FastAPI()
    app.include_router(chat_router)         # /openai/v1/...
    app.include_router(ollama_router)       # /ollama/api/...
    app.include_router(bedrock_router)      # /bedrock/model/...
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

# AWS Bedrock  (prefix: /bedrock/model)
# ---------------------------------------------------------------------------

class TestAWSBedrockRouter:
    def test_invoke_anthropic_format(self, client):
        resp = client.post(
            "/bedrock/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
        )
        assert resp.status_code == 200
        assert "content" in resp.json()

    def test_invoke_generic_format(self, client):
        resp = client.post(
            "/bedrock/model/custom-model/invoke",
            json={"prompt": "hello"},
        )
        assert resp.status_code == 200

    def test_invoke_stream(self, client):
        resp = client.post(
            "/bedrock/model/anthropic.claude-v2/invoke-with-response-stream",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_converse(self, client):
        resp = client.post(
            "/bedrock/model/anthropic.claude-3-sonnet/converse",
            json={"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
        )
        assert resp.status_code == 200
        assert "output" in resp.json()

    def test_converse_output_structure(self, client):
        resp = client.post(
            "/bedrock/model/anthropic.claude-3-sonnet/converse",
            json={"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "output" in body
        assert "message" in body["output"]
        assert body["output"]["message"]["role"] == "assistant"
        assert "content" in body["output"]["message"]
        assert body["output"]["message"]["content"][0]["text"] == "hello from fake persona"


# ---------------------------------------------------------------------------
# Ollama  (canonical prefix: /ollama/api)
# ---------------------------------------------------------------------------

