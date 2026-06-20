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

    def chat(self, messages: List[Dict], sess=None, **kwargs) -> str:
        return "hello from fake persona"

    def stream(self, messages: List[Dict], sess=None, **kwargs) -> Generator[str, None, None]:
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
    from ovos_persona_server.deprecated_routers import (
        add_deprecation_middleware,
        register_deprecated_routes,
    )

    app = FastAPI()
    app.include_router(chat_router)         # /openai/v1/...
    app.include_router(ollama_router)       # /ollama/api/...
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

# OpenAI chat  (canonical prefix: /openai/v1)
# ---------------------------------------------------------------------------

class TestOpenAIChatRouter:
    def test_chat_completions_non_stream(self, client):
        resp = client.post(
            "/openai/v1/chat/completions",
            json={"model": "fake-persona", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "hello from fake persona"

    def test_chat_completions_stream(self, client):
        resp = client.post(
            "/openai/v1/chat/completions",
            json={"model": "fake-persona", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = [l for l in resp.text.splitlines() if l.startswith("data:")]
        assert len(lines) > 1

    def test_list_models(self, client):
        resp = client.get("/openai/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert body["data"][0]["id"] == "fake-persona"

    def test_embeddings_no_solver_returns_501(self, client):
        resp = client.post(
            "/openai/v1/embeddings",
            json={"model": "text-embedding-ada-002", "input": "hello"},
        )
        assert resp.status_code == 501

    def test_chat_invalid_role_rejected(self, client):
        resp = client.post(
            "/openai/v1/chat/completions",
            json={"model": "fake-persona", "messages": [{"role": "bad_role", "content": "hi"}]},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Deprecated OpenAI paths  (/v1/... → /openai/v1/... with headers)
# ---------------------------------------------------------------------------

# Deprecated OpenAI paths  (/v1/... → /openai/v1/... with headers)
# ---------------------------------------------------------------------------

class TestDeprecatedOpenAIChatRouter:
    def test_deprecated_chat_still_works(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "fake-persona", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "hello from fake persona"

    def test_deprecated_chat_has_deprecation_header(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "fake-persona", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.headers.get("Deprecation") == "true"
        assert "/openai/v1/chat/completions" in resp.headers.get("Link", "")

    def test_deprecated_models_has_deprecation_header(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.headers.get("Deprecation") == "true"

    def test_deprecated_embeddings_still_works(self, client):
        resp = client.post(
            "/v1/embeddings",
            json={"model": "text-embedding-ada-002", "input": "hello"},
        )
        assert resp.status_code == 501  # no solver, but path is reachable


# ---------------------------------------------------------------------------
# Anthropic  (prefix: /anthropic/v1)
# ---------------------------------------------------------------------------

# Ollama  (canonical prefix: /ollama/api)
# ---------------------------------------------------------------------------

class TestOllamaRouter:
    def test_tags_returns_models_list(self, client):
        resp = client.get("/ollama/api/tags")
        assert resp.status_code == 200
        body = resp.json()
        assert "models" in body
        assert isinstance(body["models"], list)
        assert len(body["models"]) >= 1
        assert body["models"][0]["name"] == "fake-persona"

    def test_chat_returns_message_field(self, client):
        resp = client.post(
            "/ollama/api/chat",
            json={"model": "fake-persona", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "message" in body
        assert body["message"]["content"] == "hello from fake persona"
        assert body["done"] is True

    def test_show_returns_model_card(self, client):
        resp = client.get("/ollama/api/show")
        assert resp.status_code == 200
        body = resp.json()
        assert "name" in body
        assert body["name"] == "fake-persona"
        assert "details" in body

    def test_ps_returns_models_list(self, client):
        resp = client.get("/ollama/api/ps")
        assert resp.status_code == 200
        body = resp.json()
        assert "models" in body
        assert isinstance(body["models"], list)
        assert body["models"][0]["name"] == "fake-persona"

    def test_pull_returns_success_status(self, client):
        resp = client.post(
            "/ollama/api/pull",
            json={"model": "llama3"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_generate_returns_done_true(self, client):
        resp = client.post(
            "/ollama/api/generate",
            json={"model": "fake-persona", "prompt": "hello"},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Ollama /generate returns the text under "response" (not a chat "message")
        assert "response" in body
        assert body["done"] is True

    def test_embeddings_no_solver_returns_501(self, client):
        resp = client.post(
            "/ollama/api/embeddings",
            json={"model": "fake-persona", "input": "hello"},
        )
        assert resp.status_code == 501


# ---------------------------------------------------------------------------
# Deprecated Ollama paths  (/api/... → /ollama/api/... with headers)
# ---------------------------------------------------------------------------

# Deprecated Ollama paths  (/api/... → /ollama/api/... with headers)
# ---------------------------------------------------------------------------

class TestDeprecatedOllamaRouter:
    def test_deprecated_api_chat_returns_200(self, client):
        resp = client.post(
            "/api/chat",
            json={"model": "fake-persona", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200

    def test_deprecated_api_chat_has_deprecation_header(self, client):
        resp = client.post(
            "/api/chat",
            json={"model": "fake-persona", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.headers.get("Deprecation") == "true"
        link = resp.headers.get("Link", "")
        assert "/ollama/api/chat" in link
