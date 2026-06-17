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
"""End-to-end tests for the AWS Bedrock-compatible API surface.

Drives the official ``boto3`` ``bedrock-runtime`` client
(``boto3.client("bedrock-runtime", endpoint_url=...)``) against a live
ovos-persona-server instance built from ``bedrock_router`` with a mocked
Persona and served by uvicorn on a free port. The client signs requests with
SigV4 (the server ignores auth) and parses responses through botocore's
service model, so passing tests prove genuine wire compatibility — including
the binary ``vnd.amazon.eventstream`` framing for the streaming call.

Run in isolation::

    pytest tests/e2e/test_e2e_bedrock.py -v
"""
from __future__ import annotations

import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from unittest.mock import MagicMock

boto3 = pytest.importorskip("boto3", reason="boto3 not installed")
from botocore.config import Config  # noqa: E402

_CHAT_REPLY = "The capital of France is Paris."
_STREAM_CHUNKS = ["The ", "capital ", "of ", "France ", "is ", "Paris."]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app() -> FastAPI:
    from ovos_persona_server.aws_bedrock import bedrock_router
    from ovos_persona_server.persona import get_default_persona

    persona = MagicMock()
    persona.name = "test-persona"
    persona.chat.return_value = _CHAT_REPLY
    persona.stream.side_effect = lambda messages, **kwargs: iter(_STREAM_CHUNKS)

    app = FastAPI()
    app.include_router(bedrock_router)

    @app.get("/health")
    def _health() -> dict:
        return {"ok": True}

    app.dependency_overrides[get_default_persona] = lambda: persona
    return app


@pytest.fixture(scope="module")
def client():
    port = _free_port()
    config = uvicorn.Config(_build_app(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    base = f"http://127.0.0.1:{port}"
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        raise RuntimeError("server did not start in time")

    cli = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        endpoint_url=f"{base}/bedrock",
        config=Config(retries={"max_attempts": 0}),
    )
    yield cli

    server.should_exit = True
    thread.join(timeout=5)


def test_invoke_model_claude(client):
    resp = client.invoke_model(
        modelId="anthropic.claude-v2",
        body=json.dumps({
            "messages": [{"role": "user", "content": "capital of France?"}],
            "max_tokens": 64,
        }),
    )
    payload = json.loads(resp["body"].read())
    assert payload["content"][0]["text"] == _CHAT_REPLY


def test_invoke_model_titan(client):
    resp = client.invoke_model(
        modelId="amazon.titan-text-express-v1",
        body=json.dumps({"inputText": "capital of France?"}),
    )
    payload = json.loads(resp["body"].read())
    assert payload["results"][0]["outputText"] == _CHAT_REPLY


def test_converse(client):
    resp = client.converse(
        modelId="anthropic.claude-3-sonnet",
        messages=[{"role": "user", "content": [{"text": "capital of France?"}]}],
        system=[{"text": "You are a geography teacher."}],
    )
    assert resp["output"]["message"]["role"] == "assistant"
    assert resp["output"]["message"]["content"][0]["text"] == _CHAT_REPLY
    assert resp["stopReason"] == "end_turn"
    assert resp["usage"]["totalTokens"] >= resp["usage"]["outputTokens"]


def test_invoke_model_with_response_stream(client):
    resp = client.invoke_model_with_response_stream(
        modelId="amazon.titan-text-express-v1",
        body=json.dumps({"inputText": "capital of France?"}),
    )
    texts = []
    for event in resp["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if chunk.get("outputText"):
            texts.append(chunk["outputText"])
    assert "".join(texts) == "".join(_STREAM_CHUNKS)
