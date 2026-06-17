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
"""
Unit tests for the Cohere-compatible API endpoints.

Covers:
- Schema validation (missing/bad fields → 422)
- Non-streaming chat and generate
- Preamble (system message) handling
- chat_history multi-turn handling
- Streaming chat and generate
- Embed endpoint (501 when no solver)
- Unknown model/param tolerance
- Response-shape fidelity against the real Cohere API format
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeEmbedder:
    """Deterministic stand-in for the shared embeddings backend."""

    def __init__(self, vector=None):
        self.config = {"model": "fake-cohere-embed"}
        self._vector = vector if vector is not None else [0.1, 0.2, 0.3]

    def get_embeddings(self, text):
        return list(self._vector)


def _make_app(chat_returns="Hello from Cohere!", stream_yields=None, embedder="default"):
    from ovos_persona_server.cohere import cohere_router
    from ovos_persona_server.embeddings import get_embeddings_backend
    from ovos_persona_server.persona import get_default_persona

    if stream_yields is None:
        stream_yields = ["Hello ", "Cohere"]

    mock_persona = MagicMock()
    mock_persona.chat.return_value = chat_returns
    mock_persona.stream.return_value = iter(stream_yields)
    mock_persona.name = "test-persona"
    mock_persona.solvers.loaded_modules = {}

    app = FastAPI()
    app.include_router(cohere_router)
    app.dependency_overrides[get_default_persona] = lambda: mock_persona
    # Embed now goes through the shared backend; inject a fake by default so the
    # endpoint is exercised without a real embeddings plugin.
    if embedder == "default":
        app.dependency_overrides[get_embeddings_backend] = lambda: _FakeEmbedder()
    elif embedder is not None:
        app.dependency_overrides[get_embeddings_backend] = lambda: embedder
    return app, mock_persona


_VALID_CHAT_BODY = {"message": "Hello!"}
_VALID_GENERATE_BODY = {"prompt": "Tell me a joke."}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestCohereSchemaValidation:
    def setup_method(self):
        self.app, _ = _make_app()
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_chat_missing_message_422(self):
        resp = self.client.post("/cohere/v1/chat", json={})
        assert resp.status_code == 422

    def test_chat_empty_message_422(self):
        resp = self.client.post("/cohere/v1/chat", json={"message": ""})
        assert resp.status_code == 422

    def test_generate_missing_prompt_422(self):
        resp = self.client.post("/cohere/v1/generate", json={})
        assert resp.status_code == 422

    def test_generate_empty_prompt_422(self):
        resp = self.client.post("/cohere/v1/generate", json={"prompt": ""})
        assert resp.status_code == 422

    def test_embed_missing_texts_422(self):
        resp = self.client.post("/cohere/v1/embed", json={})
        assert resp.status_code == 422

    def test_embed_empty_texts_422(self):
        resp = self.client.post("/cohere/v1/embed", json={"texts": []})
        assert resp.status_code == 422

    def test_chat_invalid_temperature_422(self):
        resp = self.client.post("/cohere/v1/chat", json={**_VALID_CHAT_BODY, "temperature": 10.0})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Non-streaming chat
# ---------------------------------------------------------------------------

class TestCohereChatNonStreaming:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(chat_returns="Fine, thanks!")
        self.client = TestClient(self.app)

    def test_200_ok(self):
        resp = self.client.post("/cohere/v1/chat", json=_VALID_CHAT_BODY)
        assert resp.status_code == 200

    def test_response_has_text(self):
        resp = self.client.post("/cohere/v1/chat", json=_VALID_CHAT_BODY)
        assert resp.json()["text"] == "Fine, thanks!"

    def test_response_has_id(self):
        resp = self.client.post("/cohere/v1/chat", json=_VALID_CHAT_BODY)
        assert "id" in resp.json()

    def test_response_exact_envelope_keys(self):
        resp = self.client.post("/cohere/v1/chat", json=_VALID_CHAT_BODY)
        body = resp.json()
        for key in ("id", "text", "finish_reason", "usage"):
            assert key in body, f"Missing key: {key}"

    def test_finish_reason_complete(self):
        resp = self.client.post("/cohere/v1/chat", json=_VALID_CHAT_BODY)
        assert resp.json()["finish_reason"] == "COMPLETE"

    def test_usage_billed_units_present(self):
        resp = self.client.post("/cohere/v1/chat", json=_VALID_CHAT_BODY)
        usage = resp.json()["usage"]
        assert "billed_units" in usage

    def test_response_has_message_field(self):
        resp = self.client.post("/cohere/v1/chat", json=_VALID_CHAT_BODY)
        assert "message" in resp.json()

    def test_unknown_model_tolerated(self):
        resp = self.client.post("/cohere/v1/chat", json={**_VALID_CHAT_BODY, "model": "command-r-future"})
        assert resp.status_code == 200

    def test_extra_params_tolerated(self):
        resp = self.client.post("/cohere/v1/chat", json={**_VALID_CHAT_BODY, "temperature": 0.7})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Preamble (system message) handling
# ---------------------------------------------------------------------------

class TestCoherePreamble:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(chat_returns="Sure!")
        self.client = TestClient(self.app)

    def test_preamble_prepended_as_system(self):
        body = {**_VALID_CHAT_BODY, "preamble": "You are a pirate."}
        resp = self.client.post("/cohere/v1/chat", json=body)
        assert resp.status_code == 200
        messages = self.mock_persona.chat.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert "pirate" in messages[0]["content"]

    def test_no_preamble_no_system_message(self):
        resp = self.client.post("/cohere/v1/chat", json=_VALID_CHAT_BODY)
        messages = self.mock_persona.chat.call_args[0][0]
        roles = [m["role"] for m in messages]
        assert "system" not in roles


# ---------------------------------------------------------------------------
# Multi-turn chat_history
# ---------------------------------------------------------------------------

class TestCohereChatHistory:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(chat_returns="Sure!")
        self.client = TestClient(self.app)

    def test_chat_history_forwarded(self):
        body = {
            "message": "And 3+3?",
            "chat_history": [
                {"role": "USER", "message": "What is 2+2?"},
                {"role": "CHATBOT", "message": "4"},
            ],
        }
        resp = self.client.post("/cohere/v1/chat", json=body)
        assert resp.status_code == 200
        messages = self.mock_persona.chat.call_args[0][0]
        assert len(messages) == 3  # 2 history + 1 current

    def test_chatbot_role_mapped_to_assistant(self):
        body = {
            "message": "next",
            "chat_history": [
                {"role": "USER", "message": "hi"},
                {"role": "CHATBOT", "message": "hello"},
            ],
        }
        resp = self.client.post("/cohere/v1/chat", json=body)
        messages = self.mock_persona.chat.call_args[0][0]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1


# ---------------------------------------------------------------------------
# Streaming chat
# ---------------------------------------------------------------------------

class TestCohereChatStreaming:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(stream_yields=["chunk1 ", "chunk2"])
        self.client = TestClient(self.app)

    def test_stream_200(self):
        resp = self.client.post("/cohere/v1/chat", json={**_VALID_CHAT_BODY, "stream": True})
        assert resp.status_code == 200

    def test_stream_has_text_generation_event(self):
        resp = self.client.post("/cohere/v1/chat", json={**_VALID_CHAT_BODY, "stream": True})
        assert "text-generation" in resp.text

    def test_stream_has_stream_end_event(self):
        resp = self.client.post("/cohere/v1/chat", json={**_VALID_CHAT_BODY, "stream": True})
        assert "stream-end" in resp.text

    def test_stream_error_path(self):
        self.mock_persona.stream.side_effect = RuntimeError("stream broke")
        resp = self.client.post("/cohere/v1/chat", json={**_VALID_CHAT_BODY, "stream": True})
        assert resp.status_code == 200
        assert "ERROR" in resp.text


# ---------------------------------------------------------------------------
# Non-streaming generate
# ---------------------------------------------------------------------------

class TestCohereGenerateNonStreaming:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(chat_returns="A funny joke!")
        self.client = TestClient(self.app)

    def test_200_ok(self):
        resp = self.client.post("/cohere/v1/generate", json=_VALID_GENERATE_BODY)
        assert resp.status_code == 200

    def test_response_has_generations(self):
        resp = self.client.post("/cohere/v1/generate", json=_VALID_GENERATE_BODY)
        body = resp.json()
        assert "generations" in body
        assert len(body["generations"]) == 1

    def test_generation_text_correct(self):
        resp = self.client.post("/cohere/v1/generate", json=_VALID_GENERATE_BODY)
        assert resp.json()["generations"][0]["text"] == "A funny joke!"

    def test_generation_finish_reason(self):
        resp = self.client.post("/cohere/v1/generate", json=_VALID_GENERATE_BODY)
        assert resp.json()["generations"][0]["finish_reason"] == "COMPLETE"

    def test_response_has_id(self):
        resp = self.client.post("/cohere/v1/generate", json=_VALID_GENERATE_BODY)
        assert "id" in resp.json()


# ---------------------------------------------------------------------------
# Streaming generate
# ---------------------------------------------------------------------------

class TestCohereGenerateStreaming:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(stream_yields=["ha", "ha"])
        self.client = TestClient(self.app)

    def test_stream_200(self):
        resp = self.client.post("/cohere/v1/generate", json={**_VALID_GENERATE_BODY, "stream": True})
        assert resp.status_code == 200

    def test_stream_has_is_finished_true_at_end(self):
        resp = self.client.post("/cohere/v1/generate", json={**_VALID_GENERATE_BODY, "stream": True})
        lines = [json.loads(l) for l in resp.text.strip().splitlines() if l.strip()]
        assert any(l.get("is_finished") for l in lines)

    def test_stream_error_path(self):
        self.mock_persona.stream.side_effect = RuntimeError("boom")
        resp = self.client.post("/cohere/v1/generate", json={**_VALID_GENERATE_BODY, "stream": True})
        assert resp.status_code == 200
        assert "ERROR" in resp.text


# ---------------------------------------------------------------------------
# Embed endpoint
# ---------------------------------------------------------------------------

class TestCohereEmbed:
    def test_embed_no_backend_returns_501(self):
        from fastapi import HTTPException, status

        from ovos_persona_server.embeddings import get_embeddings_backend

        app, _ = _make_app()

        def _no_backend():
            raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED,
                                detail="No embeddings backend available")

        app.dependency_overrides[get_embeddings_backend] = _no_backend
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/cohere/v1/embed", json={"texts": ["hello", "world"]})
        assert resp.status_code == 501

    def test_embed_with_backend_returns_embeddings(self):
        app, _ = _make_app()
        client = TestClient(app)
        resp = client.post("/cohere/v1/embed", json={"texts": ["hello"]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["embeddings"] == [[0.1, 0.2, 0.3]]
        assert body["texts"] == ["hello"]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestCohereErrorPaths:
    def test_chat_persona_raises_500(self):
        app, mock_persona = _make_app()
        mock_persona.chat.side_effect = RuntimeError("backend down")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/cohere/v1/chat", json=_VALID_CHAT_BODY)
        assert resp.status_code == 500

    def test_generate_persona_raises_500(self):
        app, mock_persona = _make_app()
        mock_persona.chat.side_effect = RuntimeError("backend down")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/cohere/v1/generate", json=_VALID_GENERATE_BODY)
        assert resp.status_code == 500
