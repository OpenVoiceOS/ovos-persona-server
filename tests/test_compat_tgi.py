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
Unit tests for the HuggingFace TGI-compatible API endpoints.

Covers:
- Schema validation (missing/bad fields → 422)
- Non-streaming generate
- Streaming generate
- Info endpoint
- Health endpoint
- Unknown/extra param tolerance
- Response-shape fidelity against TGI API format
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_app(chat_returns="Generated text here.", stream_yields=None):
    from ovos_persona_server.huggingface_tgi import tgi_router
    from ovos_persona_server.persona import get_default_persona

    if stream_yields is None:
        stream_yields = ["token1 ", "token2"]

    mock_persona = MagicMock()
    mock_persona.chat.return_value = chat_returns
    mock_persona.stream.return_value = iter(stream_yields)
    mock_persona.name = "test-persona"

    app = FastAPI()
    app.include_router(tgi_router)
    app.dependency_overrides[get_default_persona] = lambda: mock_persona
    return app, mock_persona


_VALID_BODY = {"inputs": "Tell me a story."}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestTGISchemaValidation:
    def setup_method(self):
        self.app, _ = _make_app()
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_missing_inputs_422(self):
        resp = self.client.post("/tgi/generate", json={})
        assert resp.status_code == 422

    def test_empty_inputs_422(self):
        resp = self.client.post("/tgi/generate", json={"inputs": ""})
        assert resp.status_code == 422

    def test_invalid_max_new_tokens_422(self):
        resp = self.client.post("/tgi/generate", json={**_VALID_BODY, "parameters": {"max_new_tokens": 0}})
        assert resp.status_code == 422

    def test_invalid_temperature_too_high_422(self):
        resp = self.client.post("/tgi/generate", json={**_VALID_BODY, "parameters": {"temperature": 5.0}})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Non-streaming generate
# ---------------------------------------------------------------------------

class TestTGIGenerate:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(chat_returns="Once upon a time.")
        self.client = TestClient(self.app)

    def test_200_ok(self):
        resp = self.client.post("/tgi/generate", json=_VALID_BODY)
        assert resp.status_code == 200

    def test_response_has_generated_text(self):
        resp = self.client.post("/tgi/generate", json=_VALID_BODY)
        assert resp.json()["generated_text"] == "Once upon a time."

    def test_response_has_details(self):
        resp = self.client.post("/tgi/generate", json=_VALID_BODY)
        details = resp.json()["details"]
        assert details is not None
        assert "finish_reason" in details
        assert "generated_tokens" in details

    def test_details_finish_reason_eos(self):
        resp = self.client.post("/tgi/generate", json=_VALID_BODY)
        assert resp.json()["details"]["finish_reason"] == "eos_token"

    def test_exact_envelope_keys(self):
        resp = self.client.post("/tgi/generate", json=_VALID_BODY)
        body = resp.json()
        assert "generated_text" in body
        assert "details" in body

    def test_inputs_forwarded_to_persona(self):
        resp = self.client.post("/tgi/generate", json=_VALID_BODY)
        messages = self.mock_persona.chat.call_args[0][0]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Tell me a story."

    def test_unknown_params_tolerated(self):
        body = {**_VALID_BODY, "parameters": {"temperature": 0.7, "top_p": 0.9, "seed": 42}}
        resp = self.client.post("/tgi/generate", json=body)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Streaming generate
# ---------------------------------------------------------------------------

class TestTGIGenerateStream:
    def setup_method(self):
        self.app, self.mock_persona = _make_app(stream_yields=["word1 ", "word2"])
        self.client = TestClient(self.app)

    def test_stream_200(self):
        resp = self.client.post("/tgi/generate_stream", json=_VALID_BODY)
        assert resp.status_code == 200

    def test_stream_content_type(self):
        resp = self.client.post("/tgi/generate_stream", json=_VALID_BODY)
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_has_token_events(self):
        resp = self.client.post("/tgi/generate_stream", json=_VALID_BODY)
        assert "token" in resp.text

    def test_stream_final_event_has_generated_text(self):
        resp = self.client.post("/tgi/generate_stream", json=_VALID_BODY)
        # Parse all events
        events = []
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
        final = next((e for e in reversed(events) if e.get("generated_text") is not None), None)
        assert final is not None
        assert "word1" in final["generated_text"] or "word2" in final["generated_text"]

    def test_stream_error_path(self):
        self.mock_persona.stream.side_effect = RuntimeError("stream error")
        resp = self.client.post("/tgi/generate_stream", json=_VALID_BODY)
        assert resp.status_code == 200
        assert "error" in resp.text


# ---------------------------------------------------------------------------
# Info and Health endpoints
# ---------------------------------------------------------------------------

class TestTGIInfoHealth:
    def setup_method(self):
        self.app, self.mock_persona = _make_app()
        self.mock_persona.name = "my-persona"
        self.client = TestClient(self.app)

    def test_health_200(self):
        resp = self.client.get("/tgi/health")
        assert resp.status_code == 200

    def test_info_200(self):
        resp = self.client.get("/tgi/info")
        assert resp.status_code == 200

    def test_info_has_model_id(self):
        resp = self.client.get("/tgi/info")
        assert "model_id" in resp.json()

    def test_info_model_id_is_persona_name(self):
        resp = self.client.get("/tgi/info")
        assert resp.json()["model_id"] == "my-persona"

    def test_info_has_version(self):
        resp = self.client.get("/tgi/info")
        assert "version" in resp.json()

    def test_info_envelope_keys(self):
        resp = self.client.get("/tgi/info")
        body = resp.json()
        for key in ("model_id", "model_dtype", "model_device_type", "max_total_tokens", "version"):
            assert key in body, f"Missing key: {key}"
