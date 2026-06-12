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
Unit tests for the Anthropic-compatible API endpoints.

Covers:
- Schema validation (missing/bad fields → 422)
- Non-streaming single-turn and multi-turn conversations
- System message handling (prepended to messages)
- Streaming variant (SSE events)
- Unknown model name tolerance
- Response-shape fidelity against the real Anthropic API format
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_app(chat_returns="hello", stream_yields=None):
    """Build a test app with a mocked persona injected."""
    from ovos_persona_server.anthropic import anthropic_router
    from ovos_persona_server.persona import get_default_persona

    if stream_yields is None:
        stream_yields = ["hi ", "there"]

    mock_persona = MagicMock()
    mock_persona.chat.return_value = chat_returns
    mock_persona.stream.return_value = iter(stream_yields)
    mock_persona.name = "test-persona"

    app = FastAPI()
    app.include_router(anthropic_router)
    app.dependency_overrides[get_default_persona] = lambda: mock_persona
    return app, mock_persona


_VALID_BODY = {
    "model": "claude-3-opus",
    "messages": [{"role": "user", "content": "Hello"}],
}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestAnthropicSchemaValidation:
    def setup_method(self):
        self.app, _ = _make_app()
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_missing_model_422(self):
        resp = self.client.post("/anthropic/v1/messages", json={
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 422

    def test_missing_messages_422(self):
        resp = self.client.post("/anthropic/v1/messages", json={"model": "claude-3"})
        assert resp.status_code == 422

    def test_empty_messages_list_422(self):
        resp = self.client.post("/anthropic/v1/messages", json={
            "model": "claude-3", "messages": [],
        })
        assert resp.status_code == 422

    def test_invalid_role_422(self):
        resp = self.client.post("/anthropic/v1/messages", json={
            "model": "claude-3",
            "messages": [{"role": "badactor", "content": "hi"}],
        })
        assert resp.status_code == 422

    def test_zero_max_tokens_422(self):
        resp = self.client.post("/anthropic/v1/messages", json={
            "model": "claude-3",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0,
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Non-streaming responses
# ---------------------------------------------------------------------------

class TestAnthropicNonStreaming:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(chat_returns="I am Claude.")
        self.client = TestClient(self.app)

    def test_200_ok(self):
        resp = self.client.post("/anthropic/v1/messages", json=_VALID_BODY)
        assert resp.status_code == 200

    def test_response_type_field(self):
        resp = self.client.post("/anthropic/v1/messages", json=_VALID_BODY)
        assert resp.json()["type"] == "message"

    def test_response_role_is_assistant(self):
        resp = self.client.post("/anthropic/v1/messages", json=_VALID_BODY)
        assert resp.json()["role"] == "assistant"

    def test_response_has_content_list(self):
        resp = self.client.post("/anthropic/v1/messages", json=_VALID_BODY)
        body = resp.json()
        assert isinstance(body["content"], list)
        assert len(body["content"]) > 0
        assert body["content"][0]["type"] == "text"
        assert body["content"][0]["text"] == "I am Claude."

    def test_response_has_id(self):
        resp = self.client.post("/anthropic/v1/messages", json=_VALID_BODY)
        assert resp.json()["id"].startswith("msg_")

    def test_response_model_echoed(self):
        resp = self.client.post("/anthropic/v1/messages", json=_VALID_BODY)
        assert resp.json()["model"] == "claude-3-opus"

    def test_response_stop_reason(self):
        resp = self.client.post("/anthropic/v1/messages", json=_VALID_BODY)
        assert resp.json()["stop_reason"] == "end_turn"

    def test_response_usage_keys(self):
        resp = self.client.post("/anthropic/v1/messages", json=_VALID_BODY)
        usage = resp.json()["usage"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage

    def test_exact_envelope_keys(self):
        """Response must contain all required Anthropic envelope keys."""
        resp = self.client.post("/anthropic/v1/messages", json=_VALID_BODY)
        body = resp.json()
        required_keys = {"id", "type", "role", "content", "model", "stop_reason", "usage"}
        assert required_keys.issubset(body.keys())

    def test_unknown_model_tolerated(self):
        """Unknown model names must not cause errors — they are passed through."""
        resp = self.client.post("/anthropic/v1/messages", json={
            "model": "claude-999-unknown",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        assert resp.json()["model"] == "claude-999-unknown"

    def test_unknown_extra_params_tolerated(self):
        """Extra unknown fields in the request body must not cause errors."""
        resp = self.client.post("/anthropic/v1/messages", json={
            **_VALID_BODY,
            "future_param": "ignored",
            "top_k": 40,
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Multi-message conversations
# ---------------------------------------------------------------------------

class TestAnthropicMultiTurn:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(chat_returns="Sure!")
        self.client = TestClient(self.app)

    def test_multi_turn_passes_all_messages_to_persona(self):
        body = {
            "model": "claude-3",
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"},
            ],
        }
        resp = self.client.post("/anthropic/v1/messages", json=body)
        assert resp.status_code == 200
        call_args = self.mock_persona.chat.call_args[0][0]
        assert len(call_args) == 3
        assert "3+3" in call_args[-1]["content"]

    def test_content_block_list_flattened(self):
        """content as a list of AnthropicContentBlocks must be joined to string."""
        body = {
            "model": "claude-3",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]},
            ],
        }
        resp = self.client.post("/anthropic/v1/messages", json=body)
        assert resp.status_code == 200
        call_args = self.mock_persona.chat.call_args[0][0]
        assert "Hello" in call_args[0]["content"] and "World" in call_args[0]["content"]


# ---------------------------------------------------------------------------
# System message handling
# ---------------------------------------------------------------------------

class TestAnthropicSystemMessage:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(chat_returns="Yes!")
        self.client = TestClient(self.app)

    def test_system_field_prepended(self):
        body = {
            "model": "claude-3",
            "messages": [{"role": "user", "content": "Are you there?"}],
            "system": "You are a helpful assistant.",
        }
        resp = self.client.post("/anthropic/v1/messages", json=body)
        assert resp.status_code == 200
        messages = self.mock_persona.chat.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_no_system_field_no_system_message(self):
        resp = self.client.post("/anthropic/v1/messages", json=_VALID_BODY)
        messages = self.mock_persona.chat.call_args[0][0]
        assert messages[0]["role"] != "system"


# ---------------------------------------------------------------------------
# Streaming variant
# ---------------------------------------------------------------------------

class TestAnthropicStreaming:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(stream_yields=["Hello ", "World"])
        self.client = TestClient(self.app)

    def test_streaming_200(self):
        body = {**_VALID_BODY, "stream": True}
        resp = self.client.post("/anthropic/v1/messages", json=body)
        assert resp.status_code == 200

    def test_streaming_content_type(self):
        body = {**_VALID_BODY, "stream": True}
        resp = self.client.post("/anthropic/v1/messages", json=body)
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_streaming_has_message_start(self):
        body = {**_VALID_BODY, "stream": True}
        resp = self.client.post("/anthropic/v1/messages", json=body)
        assert "message_start" in resp.text

    def test_streaming_has_content_block_delta(self):
        body = {**_VALID_BODY, "stream": True}
        resp = self.client.post("/anthropic/v1/messages", json=body)
        assert "content_block_delta" in resp.text

    def test_streaming_has_message_stop(self):
        body = {**_VALID_BODY, "stream": True}
        resp = self.client.post("/anthropic/v1/messages", json=body)
        assert "message_stop" in resp.text

    def test_streaming_delta_contains_text(self):
        body = {**_VALID_BODY, "stream": True}
        resp = self.client.post("/anthropic/v1/messages", json=body)
        assert "Hello" in resp.text or "World" in resp.text

    def test_streaming_error_path(self):
        """When persona.stream raises, error event is emitted."""
        self.mock_persona.stream.side_effect = RuntimeError("stream broke")
        body = {**_VALID_BODY, "stream": True}
        resp = self.client.post("/anthropic/v1/messages", json=body)
        assert resp.status_code == 200
        assert "error" in resp.text


# ---------------------------------------------------------------------------
# API key header
# ---------------------------------------------------------------------------

class TestAnthropicAPIKey:
    def setup_method(self):
        self.app, _ = _make_app()
        self.client = TestClient(self.app)

    def test_x_api_key_accepted_and_ignored(self):
        resp = self.client.post(
            "/anthropic/v1/messages",
            json=_VALID_BODY,
            headers={"x-api-key": "sk-ant-fake-key"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestAnthropicErrorPaths:
    def test_persona_chat_exception_returns_500(self):
        app, mock_persona = _make_app()
        mock_persona.chat.side_effect = RuntimeError("backend down")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/anthropic/v1/messages", json=_VALID_BODY)
        assert resp.status_code == 500
