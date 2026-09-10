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
Regression tests: a persona whose handler chain produces no answer must not
crash the server with an ``AttributeError`` on ``None.split()``.

``ovos_persona.solvers.SolverService.chat_completion``/``stream_completion``
return ``None`` (non-streaming) or yield nothing (streaming) when every
handler in the chain declined or was never dispatched — e.g. a plugin that
loaded but was not wired into the chain. That is a legitimate, expected
outcome, so every surface that calls into ``run_chat``/``run_stream`` must
turn it into a clean error response naming the cause, never a 500 traceback
and never a silently substituted empty string.

Covers every non-streaming and streaming surface: the OpenAI chat/completions
router (``chat.py``), the legacy OpenAI text-completions path, Anthropic,
Cohere (``/chat`` and ``/generate``), Gemini, HuggingFace TGI, and AWS
Bedrock (``/invoke`` and ``/converse``).
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Non-streaming: every surface must return a clean error, not a 500 traceback
# ---------------------------------------------------------------------------

class TestNonStreamingNoAnswer:
    def _persona(self):
        mock_persona = MagicMock()
        mock_persona.chat.return_value = None
        mock_persona.name = "no-answer-persona"
        return mock_persona

    def test_openai_chat_completions(self):
        from ovos_persona_server.chat import chat_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(chat_router)
        app.dependency_overrides[get_default_persona] = self._persona
        client = TestClient(app)

        resp = client.post(
            "/openai/v1/chat/completions",
            json={"model": "no-answer-persona", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 422
        assert resp.status_code != 500
        detail = resp.json()["detail"]
        assert "no-answer-persona" in detail
        assert "no handler" in detail.lower()
        # Must not leak a raw AttributeError/traceback fragment.
        assert "nonetype" not in detail.lower()

    def test_openai_legacy_completions(self):
        from ovos_persona_server.chat import chat_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(chat_router)
        app.dependency_overrides[get_default_persona] = self._persona
        client = TestClient(app)

        resp = client.post("/openai/v1/completions", json={"model": "x", "prompt": "hi"})
        assert resp.status_code == 422
        assert resp.status_code != 500
        detail = resp.json()["detail"]
        assert "no handler" in detail.lower()
        assert "nonetype" not in detail.lower()

    def test_anthropic_messages(self):
        from ovos_persona_server.anthropic import anthropic_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(anthropic_router)
        app.dependency_overrides[get_default_persona] = self._persona
        client = TestClient(app)

        resp = client.post(
            "/anthropic/v1/messages",
            json={"model": "claude-3-opus", "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 100},
        )
        assert resp.status_code == 422
        assert resp.status_code != 500
        assert "no handler" in resp.json()["detail"].lower()

    def test_cohere_chat(self):
        from ovos_persona_server.cohere import cohere_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(cohere_router)
        app.dependency_overrides[get_default_persona] = self._persona
        client = TestClient(app)

        resp = client.post("/cohere/v1/chat", json={"message": "hi"})
        assert resp.status_code == 422
        assert resp.status_code != 500
        assert "no handler" in resp.json()["detail"].lower()

    def test_cohere_generate(self):
        from ovos_persona_server.cohere import cohere_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(cohere_router)
        app.dependency_overrides[get_default_persona] = self._persona
        client = TestClient(app)

        resp = client.post("/cohere/v1/generate", json={"prompt": "hi"})
        assert resp.status_code == 422
        assert resp.status_code != 500
        assert "no handler" in resp.json()["detail"].lower()

    def test_gemini_generate_content(self):
        from ovos_persona_server.gemini import gemini_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(gemini_router)
        app.dependency_overrides[get_default_persona] = self._persona
        client = TestClient(app)

        resp = client.post(
            "/gemini/v1beta/models/gemini-pro:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )
        assert resp.status_code == 422
        assert resp.status_code != 500
        assert "no handler" in resp.json()["detail"].lower()

    def test_tgi_generate(self):
        from ovos_persona_server.huggingface_tgi import tgi_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(tgi_router)
        app.dependency_overrides[get_default_persona] = self._persona
        client = TestClient(app)

        resp = client.post("/tgi/generate", json={"inputs": "hi"})
        assert resp.status_code == 422
        assert resp.status_code != 500
        assert "no handler" in resp.json()["detail"].lower()

    def test_bedrock_invoke(self):
        from ovos_persona_server.aws_bedrock import bedrock_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(bedrock_router)
        app.dependency_overrides[get_default_persona] = self._persona
        client = TestClient(app)

        resp = client.post(
            "/bedrock/model/anthropic.claude-v2/invoke",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 422
        assert resp.status_code != 500
        assert "no handler" in resp.json()["detail"].lower()

    def test_bedrock_converse(self):
        from ovos_persona_server.aws_bedrock import bedrock_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(bedrock_router)
        app.dependency_overrides[get_default_persona] = self._persona
        client = TestClient(app)

        resp = client.post(
            "/bedrock/model/anthropic.claude-v2/converse",
            json={"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
        )
        assert resp.status_code == 422
        assert resp.status_code != 500
        assert "no handler" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Streaming: the HTTP status is already committed to 200 by the time the
# stream starts, so a no-answer outcome has to surface in-band as an SSE/NDJSON
# error event rather than a status code. The client sees a normal 200 stream
# whose only payload is an error event (no content chunks, no clean [DONE]).
# ---------------------------------------------------------------------------

class TestStreamingNoAnswer:
    def _persona(self):
        mock_persona = MagicMock()
        mock_persona.stream.return_value = iter([])  # no handler ever yielded a token
        mock_persona.name = "no-answer-persona"
        return mock_persona

    def test_openai_chat_completions_stream(self):
        from ovos_persona_server.chat import chat_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(chat_router)
        app.dependency_overrides[get_default_persona] = self._persona
        client = TestClient(app)

        resp = client.post(
            "/openai/v1/chat/completions",
            json={"model": "no-answer-persona", "messages": [{"role": "user", "content": "hi"}],
                  "stream": True},
        )
        assert resp.status_code == 200  # status already committed before the stream starts
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "no handler" in body.lower()
        # No content chunk and no clean [DONE] sentinel — the stream stops at the error.
        assert '"content": "' not in body or '"content": ""' in body
        assert "[DONE]" not in body

        # The real ``openai`` SDK's streaming parser only surfaces a message
        # when ``error`` is a mapping with a "message" key; a bare string is
        # discarded and replaced with a generic ``APIError``. Parse the SSE
        # payload the same way the SDK does to prove the diagnostic survives.
        import json as _json
        error_events = [
            _json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: ") and '"error"' in line
        ]
        assert error_events, "expected an SSE event carrying an error object"
        error_obj = error_events[0]["error"]
        assert isinstance(error_obj, dict)
        assert "no handler" in error_obj["message"].lower()

    def test_anthropic_messages_stream(self):
        from ovos_persona_server.anthropic import anthropic_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(anthropic_router)
        app.dependency_overrides[get_default_persona] = self._persona
        client = TestClient(app)

        resp = client.post(
            "/anthropic/v1/messages",
            json={"model": "claude-3-opus", "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 100, "stream": True},
        )
        assert resp.status_code == 200
        body = resp.text
        assert "event: error" in body
        assert "no handler" in body.lower()
        assert "message_stop" not in body

    def test_cohere_chat_stream(self):
        from ovos_persona_server.cohere import cohere_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(cohere_router)
        app.dependency_overrides[get_default_persona] = self._persona
        client = TestClient(app)

        resp = client.post("/cohere/v1/chat", json={"message": "hi", "stream": True})
        assert resp.status_code == 200
        body = resp.text
        assert '"finish_reason": "ERROR"' in body
        assert "no handler" in body.lower()

    def test_gemini_stream_generate_content(self):
        from ovos_persona_server.gemini import gemini_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(gemini_router)
        app.dependency_overrides[get_default_persona] = self._persona
        client = TestClient(app)

        resp = client.post(
            "/gemini/v1beta/models/gemini-pro:streamGenerateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )
        assert resp.status_code == 200
        body = resp.text
        assert '"error"' in body
        assert "no handler" in body.lower()

    def test_tgi_generate_stream(self):
        from ovos_persona_server.huggingface_tgi import tgi_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(tgi_router)
        app.dependency_overrides[get_default_persona] = self._persona
        client = TestClient(app)

        resp = client.post("/tgi/generate_stream", json={"inputs": "hi"})
        assert resp.status_code == 200
        body = resp.text
        assert '"error"' in body
        assert "no handler" in body.lower()

    def test_bedrock_invoke_stream(self):
        from ovos_persona_server.aws_bedrock import bedrock_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(bedrock_router)
        app.dependency_overrides[get_default_persona] = self._persona
        client = TestClient(app)

        resp = client.post(
            "/bedrock/model/anthropic.claude-v2/invoke-with-response-stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        # AWS event-stream framing is binary and each frame's "bytes" field is
        # itself base64-encoded JSON. Decode every embedded base64 run and
        # confirm the error message survived the round trip.
        import base64
        import json
        import re

        decoded_texts = []
        for match in re.finditer(rb"[A-Za-z0-9+/]{20,}={0,2}", resp.content):
            try:
                decoded_texts.append(base64.b64decode(match.group()).decode("utf-8"))
            except Exception:
                continue
        assert any("no handler" in json.dumps(json.loads(t)).lower() for t in decoded_texts)


# ---------------------------------------------------------------------------
# Ollama: unlike every other vendor router, ``ollama.py`` used to call
# ``persona.chat``/``persona.stream`` directly, bypassing ``run_chat``/
# ``run_stream`` entirely — so a no-answer persona returned a raw pydantic
# validation error (500) non-streaming, and a false-success ``"done": true``
# with no error at all when streaming.
# ---------------------------------------------------------------------------

class TestOllamaNoAnswer:
    def _chat_persona(self):
        mock_persona = MagicMock()
        mock_persona.chat.return_value = None
        mock_persona.name = "no-answer-persona"
        return mock_persona

    def _stream_persona(self):
        mock_persona = MagicMock()
        mock_persona.stream.return_value = iter([])
        mock_persona.name = "no-answer-persona"
        return mock_persona

    def test_ollama_chat(self):
        from ovos_persona_server.ollama import ollama_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(ollama_router)
        app.dependency_overrides[get_default_persona] = self._chat_persona
        client = TestClient(app)

        resp = client.post(
            "/ollama/api/chat",
            json={"model": "no-answer-persona", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 422
        assert resp.status_code != 500
        detail = resp.json()["detail"]
        assert "no handler" in detail.lower()

    def test_ollama_chat_stream(self):
        from ovos_persona_server.ollama import ollama_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(ollama_router)
        app.dependency_overrides[get_default_persona] = self._stream_persona
        client = TestClient(app)

        resp = client.post(
            "/ollama/api/chat",
            json={"model": "no-answer-persona", "messages": [{"role": "user", "content": "hi"}],
                  "stream": True},
        )
        assert resp.status_code == 200
        lines = [json.loads(l) for l in resp.text.splitlines() if l.strip()]
        assert lines, "expected at least one NDJSON line"
        # A no-answer outcome must be reported as an error, never a bare
        # "done": true success — the false-success marker the old code emitted
        # when it called persona.stream() directly and swallowed the no-yield case.
        assert any("error" in l for l in lines)
        assert "no handler" in json.dumps(lines).lower()

    def test_ollama_generate(self):
        from ovos_persona_server.ollama import ollama_router
        from ovos_persona_server.persona import get_default_persona

        app = FastAPI()
        app.include_router(ollama_router)
        app.dependency_overrides[get_default_persona] = self._chat_persona
        client = TestClient(app)

        resp = client.post(
            "/ollama/api/generate",
            json={"model": "no-answer-persona", "prompt": "hi"},
        )
        assert resp.status_code == 422
        assert resp.status_code != 500
        detail = resp.json()["detail"]
        assert "no handler" in detail.lower()
