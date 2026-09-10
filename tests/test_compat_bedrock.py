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
Unit tests for the AWS Bedrock-compatible API endpoints.

Covers:
- _extract_messages for all supported model families (Anthropic, Llama, Titan, Cohere, Converse)
- _build_response shape for each model family
- /invoke endpoint with various model IDs
- /invoke-with-response-stream streaming endpoint
- /converse endpoint
- Schema validation on /converse (missing required fields → 422)
- Unknown model response (generic fallback)
- Multi-turn messages via Converse API
- System message via Converse API
- Response-shape fidelity per model family
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ovos_plugin_manager.templates.agents import MessageRole


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_app(chat_returns="Bedrock reply", stream_yields=None):
    from ovos_persona_server.aws_bedrock import bedrock_router
    from ovos_persona_server.persona import get_default_persona

    if stream_yields is None:
        stream_yields = ["chunk1 ", "chunk2"]

    mock_persona = MagicMock()
    mock_persona.chat.return_value = chat_returns
    mock_persona.stream.return_value = iter(stream_yields)
    mock_persona.name = "test-persona"

    app = FastAPI()
    app.include_router(bedrock_router)
    app.dependency_overrides[get_default_persona] = lambda: mock_persona
    return app, mock_persona


# ---------------------------------------------------------------------------
# _extract_messages unit tests
# ---------------------------------------------------------------------------

class TestExtractMessages:
    def test_converse_format_basic(self):
        from ovos_persona_server.aws_bedrock import _extract_messages
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        result = _extract_messages(body, "any-model")
        assert result == [{"role": "user", "content": "Hello"}]

    def test_converse_format_content_block_list(self):
        from ovos_persona_server.aws_bedrock import _extract_messages
        body = {"messages": [{"role": "user", "content": [{"text": "Hello"}, {"text": "World"}]}]}
        result = _extract_messages(body, "any-model")
        assert "Hello" in result[0]["content"] and "World" in result[0]["content"]

    def test_converse_with_system(self):
        from ovos_persona_server.aws_bedrock import _extract_messages
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "system": [{"text": "Be helpful."}],
        }
        result = _extract_messages(body, "any-model")
        assert result[0]["role"] == "system"
        assert "helpful" in result[0]["content"]

    def test_converse_system_string(self):
        from ovos_persona_server.aws_bedrock import _extract_messages
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "system": "You are a bot.",
        }
        result = _extract_messages(body, "any-model")
        assert result[0]["role"] == "system"

    def test_prompt_format(self):
        from ovos_persona_server.aws_bedrock import _extract_messages
        body = {"prompt": "What is AI?"}
        result = _extract_messages(body, "meta.llama2")
        assert result == [{"role": "user", "content": "What is AI?"}]

    def test_input_text_format(self):
        from ovos_persona_server.aws_bedrock import _extract_messages
        body = {"inputText": "Explain quantum physics."}
        result = _extract_messages(body, "amazon.titan-text")
        assert result == [{"role": "user", "content": "Explain quantum physics."}]

    def test_cohere_message_format(self):
        from ovos_persona_server.aws_bedrock import _extract_messages
        body = {
            "message": "What is the capital of France?",
            "chat_history": [
                {"role": "USER", "message": "Hello"},
                {"role": "CHATBOT", "message": "Hi!"},
            ],
        }
        result = _extract_messages(body, "cohere.command-r")
        assert result[-1]["role"] == "user"
        assert result[-1]["content"] == "What is the capital of France?"
        assert len(result) == 3

    def test_unknown_format_fallback(self):
        from ovos_persona_server.aws_bedrock import _extract_messages
        body = {"totally_unknown": "field"}
        result = _extract_messages(body, "unknown-model")
        assert len(result) == 1
        assert result[0]["role"] == "user"


# ---------------------------------------------------------------------------
# _build_response unit tests
# ---------------------------------------------------------------------------

class TestBuildResponse:
    def test_anthropic_claude_format(self):
        from ovos_persona_server.aws_bedrock import _build_response
        resp = _build_response("Hello!", "anthropic.claude-3-opus-20240229-v1:0", {})
        assert resp["type"] == "message"
        assert resp["role"] == "assistant"
        assert resp["content"][0]["text"] == "Hello!"
        assert resp["stop_reason"] == "end_turn"

    def test_meta_llama_format(self):
        from ovos_persona_server.aws_bedrock import _build_response
        resp = _build_response("Llama reply", "meta.llama2-13b-chat-v1", {})
        assert "generation" in resp
        assert resp["generation"] == "Llama reply"
        assert resp["stop_reason"] == "stop"

    def test_amazon_titan_format(self):
        from ovos_persona_server.aws_bedrock import _build_response
        resp = _build_response("Titan text", "amazon.titan-text-express-v1", {})
        assert "results" in resp
        assert resp["results"][0]["outputText"] == "Titan text"

    def test_cohere_command_format(self):
        from ovos_persona_server.aws_bedrock import _build_response
        resp = _build_response("Cohere gen", "cohere.command-r-v1:0", {})
        assert "generations" in resp
        assert resp["generations"][0]["text"] == "Cohere gen"

    def test_generic_fallback_format(self):
        from ovos_persona_server.aws_bedrock import _build_response
        resp = _build_response("Generic", "some-other-model", {})
        assert "outputText" in resp
        assert resp["outputText"] == "Generic"


# ---------------------------------------------------------------------------
# /invoke endpoint
# ---------------------------------------------------------------------------

class TestBedrockInvoke:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(chat_returns="Invoking Bedrock!")
        self.client = TestClient(self.app)

    def _invoke(self, model_id, body):
        return self.client.post(f"/bedrock/model/{model_id}/invoke", json=body)

    def test_invoke_anthropic_200(self):
        resp = self._invoke("anthropic.claude-3", {"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        assert body["content"][0]["text"] == "Invoking Bedrock!"

    def test_invoke_llama_200(self):
        resp = self._invoke("meta.llama2-13b", {"prompt": "Tell me a story."})
        assert resp.status_code == 200
        assert "generation" in resp.json()

    def test_invoke_titan_200(self):
        resp = self._invoke("amazon.titan-text-v1", {"inputText": "Summarise."})
        assert resp.status_code == 200
        assert "results" in resp.json()

    def test_invoke_cohere_200(self):
        resp = self._invoke("cohere.command-r", {"message": "What is AI?"})
        assert resp.status_code == 200
        assert "generations" in resp.json()

    def test_invoke_unknown_model_generic_fallback(self):
        resp = self._invoke("some-custom-model", {"prompt": "test"})
        assert resp.status_code == 200
        assert "outputText" in resp.json()

    def test_auth_header_accepted_and_ignored(self):
        resp = self.client.post(
            "/bedrock/model/anthropic.claude-3/invoke",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "AWS4-HMAC-SHA256 fake-sig"},
        )
        assert resp.status_code == 200

    def test_multi_turn_converse_format(self):
        body = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"},
            ],
        }
        resp = self._invoke("anthropic.claude-3", body)
        assert resp.status_code == 200
        call_msgs = self.mock_persona.chat.call_args[0][0]
        assert len(call_msgs) == 3


# ---------------------------------------------------------------------------
# /invoke-with-response-stream endpoint
# ---------------------------------------------------------------------------

def _decode_eventstream(content: bytes) -> list:
    """Decode vnd.amazon.eventstream bytes into model-specific chunk dicts.

    Uses botocore's own decoder so the test verifies the exact binary framing
    boto3 consumes (prelude/headers/payload with CRC32 checksums), then unwraps
    the base64-encoded chunk each event carries under the ``bytes`` key.
    """
    import base64

    from botocore.eventstream import EventStreamBuffer

    buffer = EventStreamBuffer()
    buffer.add_data(content)
    chunks = []
    for message in buffer:
        payload = json.loads(message.payload)
        chunks.append(json.loads(base64.b64decode(payload["bytes"])))
    return chunks


class TestBedrockInvokeStream:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(stream_yields=["chunk1 ", "chunk2"])
        self.client = TestClient(self.app)

    def test_stream_200(self):
        resp = self.client.post(
            "/bedrock/model/anthropic.claude-3/invoke-with-response-stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200

    def test_stream_content_type(self):
        resp = self.client.post(
            "/bedrock/model/anthropic.claude-3/invoke-with-response-stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert "application/vnd.amazon.eventstream" in resp.headers.get("content-type", "")

    def test_stream_has_output_text(self):
        resp = self.client.post(
            "/bedrock/model/anthropic.claude-3/invoke-with-response-stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        chunks = _decode_eventstream(resp.content)
        assert any("outputText" in c for c in chunks)

    def test_stream_final_event_has_completion_reason(self):
        resp = self.client.post(
            "/bedrock/model/anthropic.claude-3/invoke-with-response-stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        chunks = _decode_eventstream(resp.content)
        final = next((c for c in reversed(chunks) if c.get("completionReason")), None)
        assert final is not None
        assert final["completionReason"] == "FINISH"

    def test_stream_error_path(self):
        self.mock_persona.stream.side_effect = RuntimeError("stream boom")
        resp = self.client.post(
            "/bedrock/model/anthropic.claude-3/invoke-with-response-stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        chunks = _decode_eventstream(resp.content)
        assert any("error" in c for c in chunks)


# ---------------------------------------------------------------------------
# /converse endpoint
# ---------------------------------------------------------------------------

class TestBedrockConverse:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(chat_returns="Converse reply")
        self.client = TestClient(self.app)

    def _converse(self, model_id, body):
        return self.client.post(f"/bedrock/model/{model_id}/converse", json=body)

    def test_converse_200(self):
        resp = self._converse("anthropic.claude-3", {
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
        })
        assert resp.status_code == 200

    def test_converse_response_shape(self):
        resp = self._converse("anthropic.claude-3", {
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
        })
        body = resp.json()
        assert "output" in body
        assert "stopReason" in body
        assert "usage" in body

    def test_converse_output_message(self):
        resp = self._converse("any-model", {
            "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
        })
        msg = resp.json()["output"]["message"]
        assert msg["role"] == "assistant"
        assert msg["content"][0]["text"] == "Converse reply"

    def test_converse_stop_reason_end_turn(self):
        resp = self._converse("anthropic.claude-3", {
            "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
        })
        assert resp.json()["stopReason"] == "end_turn"

    def test_converse_usage_keys(self):
        resp = self._converse("anthropic.claude-3", {
            "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
        })
        usage = resp.json()["usage"]
        for key in ("inputTokens", "outputTokens", "totalTokens"):
            assert key in usage, f"Missing key: {key}"

    def test_converse_missing_messages_422(self):
        resp = self._converse("anthropic.claude-3", {})
        assert resp.status_code == 422

    def test_converse_empty_messages_422(self):
        resp = self._converse("anthropic.claude-3", {"messages": []})
        assert resp.status_code == 422

    def test_converse_system_field(self):
        resp = self._converse("anthropic.claude-3", {
            "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
            "system": [{"text": "You are helpful."}],
        })
        assert resp.status_code == 200
        msgs = self.mock_persona.chat.call_args[0][0]
        assert msgs[0].role == MessageRole.SYSTEM

    def test_converse_multi_turn(self):
        resp = self._converse("anthropic.claude-3", {
            "messages": [
                {"role": "user", "content": [{"text": "Hi"}]},
                {"role": "assistant", "content": [{"text": "Hello"}]},
                {"role": "user", "content": [{"text": "How are you?"}]},
            ],
        })
        assert resp.status_code == 200
        msgs = self.mock_persona.chat.call_args[0][0]
        assert len(msgs) == 3
