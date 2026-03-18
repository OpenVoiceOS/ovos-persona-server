# Licensed under the Apache License, Version 2.0
"""Unit tests for persona server compatibility routers.

Uses dependency-override to replace the live Persona with a lightweight stub.
"""
import json
from typing import Dict, Generator, List, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fake persona
# ---------------------------------------------------------------------------

class FakePersona:
    """Minimal Persona stub returning deterministic responses."""

    name: str = "fake-persona"

    def chat(self, messages: List[Dict]) -> str:
        return "hello from fake persona"

    def stream(self, messages: List[Dict]) -> Generator[str, None, None]:
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
    from ovos_persona_server.anthropic import anthropic_router
    from ovos_persona_server.gemini import gemini_router
    from ovos_persona_server.cohere import cohere_router
    from ovos_persona_server.huggingface_tgi import tgi_router
    from ovos_persona_server.aws_bedrock import bedrock_router

    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(anthropic_router)
    app.include_router(gemini_router)
    app.include_router(cohere_router)
    app.include_router(tgi_router)
    app.include_router(bedrock_router)
    app.dependency_overrides[get_default_persona] = _override_persona
    return app


@pytest.fixture(scope="module")
def client():
    app = _make_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# OpenAI chat (existing chat_router)
# ---------------------------------------------------------------------------

class TestOpenAIChatRouter:
    def test_chat_completions_non_stream(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "fake-persona", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "hello from fake persona"
        assert body["object"] == "chat.completion"

    def test_chat_completions_stream(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "fake-persona", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = [l for l in resp.text.splitlines() if l.startswith("data:")]
        assert len(lines) > 1

    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert body["data"][0]["id"] == "fake-persona"

    def test_embeddings_no_solver_returns_501(self, client):
        resp = client.post(
            "/v1/embeddings",
            json={"model": "text-embedding-ada-002", "input": "hello"},
        )
        assert resp.status_code == 501

    def test_chat_invalid_role_rejected(self, client):
        """An invalid role should fail Pydantic validation."""
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "fake-persona", "messages": [{"role": "invalid_role", "content": "hi"}]},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class TestAnthropicRouter:
    def test_create_message_non_stream(self, client):
        resp = client.post(
            "/v1/messages",
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
        assert body["role"] == "assistant"
        assert body["content"][0]["text"] == "hello from fake persona"

    def test_create_message_stream(self, client):
        resp = client.post(
            "/v1/messages",
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

    def test_system_message_prepended(self, client):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": "hi"}],
                "system": "You are helpful.",
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 200

    def test_empty_messages_rejected(self, client):
        resp = client.post(
            "/v1/messages",
            json={"model": "claude-3-opus-20240229", "messages": [], "max_tokens": 100},
        )
        assert resp.status_code == 422

    def test_zero_max_tokens_rejected(self, client):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 0,
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class TestGeminiRouter:
    def test_generate_content(self, client):
        resp = client.post(
            "/v1beta/models/gemini-pro:generateContent?key=fake",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "candidates" in body
        assert body["candidates"][0]["content"]["parts"][0]["text"] == "hello from fake persona"

    def test_stream_generate_content(self, client):
        resp = client.post(
            "/v1beta/models/gemini-pro:streamGenerateContent?key=fake",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_empty_contents_rejected(self, client):
        resp = client.post(
            "/v1beta/models/gemini-pro:generateContent",
            json={"contents": []},
        )
        assert resp.status_code == 422

    def test_model_role_accepted(self, client):
        resp = client.post(
            "/v1beta/models/gemini-pro:generateContent",
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
# Cohere
# ---------------------------------------------------------------------------

class TestCohereRouter:
    def test_chat_non_stream(self, client):
        resp = client.post(
            "/v1/chat",
            json={"message": "hello"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body
        assert body["text"] == "hello from fake persona"

    def test_chat_stream(self, client):
        resp = client.post(
            "/v1/chat",
            json={"message": "hello", "stream": True},
        )
        assert resp.status_code == 200
        lines = [l for l in resp.text.splitlines() if l.strip()]
        events = [json.loads(l) for l in lines if l.startswith("{")]
        event_types = [e.get("event_type") for e in events]
        assert "stream-end" in event_types

    def test_chat_empty_message_rejected(self, client):
        resp = client.post("/v1/chat", json={"message": ""})
        assert resp.status_code == 422

    def test_generate_non_stream(self, client):
        resp = client.post(
            "/v1/generate",
            json={"prompt": "once upon a time"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "generations" in body

    def test_generate_empty_prompt_rejected(self, client):
        resp = client.post("/v1/generate", json={"prompt": ""})
        assert resp.status_code == 422

    def test_temperature_out_of_range_rejected(self, client):
        resp = client.post("/v1/chat", json={"message": "hi", "temperature": 6.0})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# HuggingFace TGI
# ---------------------------------------------------------------------------

class TestTGIRouter:
    def test_generate_basic(self, client):
        resp = client.post(
            "/generate",
            json={"inputs": "once upon a time"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "generated_text" in body
        assert body["generated_text"] == "hello from fake persona"

    def test_generate_empty_inputs_rejected(self, client):
        resp = client.post("/generate", json={"inputs": ""})
        assert resp.status_code == 422

    def test_generate_stream(self, client):
        resp = client.post(
            "/generate_stream",
            json={"inputs": "hello"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_info(self, client):
        resp = client.get("/info")
        assert resp.status_code == 200
        body = resp.json()
        assert "model_id" in body

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# AWS Bedrock
# ---------------------------------------------------------------------------

class TestAWSBedrockRouter:
    def test_invoke_anthropic_format(self, client):
        resp = client.post(
            "/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        # Anthropic family response
        assert "content" in body

    def test_invoke_generic_format(self, client):
        resp = client.post(
            "/model/custom-model/invoke",
            json={"prompt": "hello"},
        )
        assert resp.status_code == 200

    def test_invoke_stream(self, client):
        resp = client.post(
            "/model/anthropic.claude-v2/invoke-with-response-stream",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_converse(self, client):
        resp = client.post(
            "/model/anthropic.claude-3-sonnet/converse",
            json={"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "output" in body
