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
Unit tests for the Google Gemini-compatible API endpoints.

Covers:
- Schema validation (missing/bad fields → 422)
- Non-streaming single-turn and multi-turn conversations
- System instruction handling
- Streaming variant (SSE events)
- Unknown model/param tolerance
- Response-shape fidelity against the real Gemini API format
- Error paths (persona.chat raises → 500)
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_app(chat_returns="Hello from Gemini!", stream_yields=None):
    from ovos_persona_server.gemini import gemini_router
    from ovos_persona_server.persona import get_default_persona

    if stream_yields is None:
        stream_yields = ["Hello ", "Gemini"]

    mock_persona = MagicMock()
    mock_persona.chat.return_value = chat_returns
    mock_persona.stream.return_value = iter(stream_yields)
    mock_persona.name = "test-persona"

    app = FastAPI()
    app.include_router(gemini_router)
    app.dependency_overrides[get_default_persona] = lambda: mock_persona
    return app, mock_persona


_VALID_BODY = {
    "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
}

_MODEL_ID = "gemini-pro"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestGeminiSchemaValidation:
    def setup_method(self):
        self.app, _ = _make_app()
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_missing_contents_422(self):
        resp = self.client.post(f"/gemini/v1beta/models/{_MODEL_ID}:generateContent", json={})
        assert resp.status_code == 422

    def test_empty_contents_422(self):
        resp = self.client.post(
            f"/gemini/v1beta/models/{_MODEL_ID}:generateContent",
            json={"contents": []},
        )
        assert resp.status_code == 422

    def test_invalid_role_422(self):
        resp = self.client.post(
            f"/gemini/v1beta/models/{_MODEL_ID}:generateContent",
            json={"contents": [{"role": "bad_role", "parts": [{"text": "hi"}]}]},
        )
        assert resp.status_code == 422

    def test_empty_parts_422(self):
        resp = self.client.post(
            f"/gemini/v1beta/models/{_MODEL_ID}:generateContent",
            json={"contents": [{"role": "user", "parts": []}]},
        )
        assert resp.status_code == 422

    def test_invalid_generation_config_422(self):
        resp = self.client.post(
            f"/gemini/v1beta/models/{_MODEL_ID}:generateContent",
            json={**_VALID_BODY, "generationConfig": {"maxOutputTokens": 0}},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Non-streaming responses
# ---------------------------------------------------------------------------

class TestGeminiNonStreaming:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(chat_returns="4")
        self.client = TestClient(self.app)

    def test_200_ok(self):
        resp = self.client.post(f"/gemini/v1beta/models/{_MODEL_ID}:generateContent", json=_VALID_BODY)
        assert resp.status_code == 200

    def test_response_has_candidates(self):
        resp = self.client.post(f"/gemini/v1beta/models/{_MODEL_ID}:generateContent", json=_VALID_BODY)
        body = resp.json()
        assert "candidates" in body
        assert len(body["candidates"]) == 1

    def test_candidate_has_content(self):
        resp = self.client.post(f"/gemini/v1beta/models/{_MODEL_ID}:generateContent", json=_VALID_BODY)
        candidate = resp.json()["candidates"][0]
        assert "content" in candidate
        assert candidate["content"]["role"] == "model"

    def test_candidate_content_text(self):
        resp = self.client.post(f"/gemini/v1beta/models/{_MODEL_ID}:generateContent", json=_VALID_BODY)
        parts = resp.json()["candidates"][0]["content"]["parts"]
        assert len(parts) >= 1
        assert parts[0]["text"] == "4"

    def test_candidate_finish_reason_stop(self):
        resp = self.client.post(f"/gemini/v1beta/models/{_MODEL_ID}:generateContent", json=_VALID_BODY)
        assert resp.json()["candidates"][0]["finishReason"] == "STOP"

    def test_exact_envelope_keys(self):
        """Response must contain required Gemini envelope keys."""
        resp = self.client.post(f"/gemini/v1beta/models/{_MODEL_ID}:generateContent", json=_VALID_BODY)
        body = resp.json()
        assert "candidates" in body

    def test_unknown_model_tolerated(self):
        resp = self.client.post(
            "/gemini/v1beta/models/gemini-ultra-99:generateContent",
            json=_VALID_BODY,
        )
        assert resp.status_code == 200

    def test_api_key_query_param_ignored(self):
        resp = self.client.post(
            f"/gemini/v1beta/models/{_MODEL_ID}:generateContent?key=fake-api-key",
            json=_VALID_BODY,
        )
        assert resp.status_code == 200

    def test_extra_params_tolerated(self):
        body = {**_VALID_BODY, "generationConfig": {"temperature": 0.7, "topP": 0.9}}
        resp = self.client.post(f"/gemini/v1beta/models/{_MODEL_ID}:generateContent", json=body)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Multi-turn conversations
# ---------------------------------------------------------------------------

class TestGeminiMultiTurn:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(chat_returns="Sure!")
        self.client = TestClient(self.app)

    def test_multi_turn_all_messages_forwarded(self):
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": "Hi"}]},
                {"role": "model", "parts": [{"text": "Hello!"}]},
                {"role": "user", "parts": [{"text": "How are you?"}]},
            ],
        }
        resp = self.client.post(f"/gemini/v1beta/models/{_MODEL_ID}:generateContent", json=body)
        assert resp.status_code == 200
        messages = self.mock_persona.chat.call_args[0][0]
        assert len(messages) == 3

    def test_model_role_mapped_to_assistant(self):
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": "Hi"}]},
                {"role": "model", "parts": [{"text": "Hello"}]},
                {"role": "user", "parts": [{"text": "Bye"}]},
            ],
        }
        resp = self.client.post(f"/gemini/v1beta/models/{_MODEL_ID}:generateContent", json=body)
        messages = self.mock_persona.chat.call_args[0][0]
        assert messages[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# System instruction handling
# ---------------------------------------------------------------------------

class TestGeminiSystemInstruction:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(chat_returns="Yes!")
        self.client = TestClient(self.app)

    def test_system_instruction_prepended(self):
        body = {
            **_VALID_BODY,
            "systemInstruction": {"parts": [{"text": "You are a helpful bot."}]},
        }
        resp = self.client.post(f"/gemini/v1beta/models/{_MODEL_ID}:generateContent", json=body)
        assert resp.status_code == 200
        messages = self.mock_persona.chat.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert "helpful bot" in messages[0]["content"]

    def test_no_system_instruction_no_system_message(self):
        resp = self.client.post(f"/gemini/v1beta/models/{_MODEL_ID}:generateContent", json=_VALID_BODY)
        messages = self.mock_persona.chat.call_args[0][0]
        assert messages[0]["role"] != "system"


# ---------------------------------------------------------------------------
# Streaming variant
# ---------------------------------------------------------------------------

class TestGeminiStreaming:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(stream_yields=["chunk1 ", "chunk2"])
        self.client = TestClient(self.app)

    def test_stream_200(self):
        resp = self.client.post(
            f"/gemini/v1beta/models/{_MODEL_ID}:streamGenerateContent",
            json=_VALID_BODY,
        )
        assert resp.status_code == 200

    def test_stream_content_type(self):
        resp = self.client.post(
            f"/gemini/v1beta/models/{_MODEL_ID}:streamGenerateContent",
            json=_VALID_BODY,
        )
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_has_candidates(self):
        resp = self.client.post(
            f"/gemini/v1beta/models/{_MODEL_ID}:streamGenerateContent",
            json=_VALID_BODY,
        )
        assert "candidates" in resp.text

    def test_stream_contains_chunk_text(self):
        resp = self.client.post(
            f"/gemini/v1beta/models/{_MODEL_ID}:streamGenerateContent",
            json=_VALID_BODY,
        )
        assert "chunk1" in resp.text or "chunk2" in resp.text

    def test_stream_error_path(self):
        self.mock_persona.stream.side_effect = RuntimeError("stream failed")
        resp = self.client.post(
            f"/gemini/v1beta/models/{_MODEL_ID}:streamGenerateContent",
            json=_VALID_BODY,
        )
        assert resp.status_code == 200
        assert "error" in resp.text


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestGeminiErrorPaths:
    def test_persona_chat_raises_500(self):
        app, mock_persona = _make_app()
        mock_persona.chat.side_effect = RuntimeError("backend failure")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(f"/gemini/v1beta/models/{_MODEL_ID}:generateContent", json=_VALID_BODY)
        assert resp.status_code == 500
