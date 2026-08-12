"""Tests for the OpenAI-backed tools passthrough proxy.

When a chat-completions request carries a non-empty ``tools`` list and the
persona's primary solver is an OpenAI-compatible chat endpoint, the request is
proxied verbatim to that endpoint and its reply (tool_calls, finish_reason,
streamed deltas) is relayed to the client. A tiny local HTTP stub stands in for
the upstream OpenAI endpoint so the relay behaviour can be asserted end to end.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]

_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-upstream",
    "object": "chat.completion",
    "created": 1,
    "model": "upstream-model",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Lisbon"}'},
            }],
        },
        "finish_reason": "tool_calls",
    }],
    "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
}

_STREAM_LINES = [
    'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,'
    '"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_abc",'
    '"type":"function","function":{"name":"get_weather","arguments":""}}]},'
    '"finish_reason":null}]}',
    'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,'
    '"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\":\\"Lisbon\\"}"}}]},'
    '"finish_reason":null}]}',
    'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,'
    '"delta":{},"finish_reason":"tool_calls"}]}',
    'data: [DONE]',
]


class _StubUpstream:
    """A local HTTP server standing in for an OpenAI chat endpoint."""

    def __init__(self, mode="tool_calls", delay=0.0):
        self.mode = mode
        self.delay = delay
        self.captured = []  # request payloads seen
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                stub.captured.append(body)
                if stub.delay:
                    time.sleep(stub.delay)
                if stub.mode == "error":
                    payload = json.dumps({"error": {"message": "boom", "type": "server_error"}}).encode()
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for line in _STREAM_LINES:
                        self.wfile.write((line + "\n\n").encode())
                        self.wfile.flush()
                    return
                payload = json.dumps(_TOOL_CALL_RESPONSE).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._server.shutdown()
        self._server.server_close()

    @property
    def chat_url(self):
        return f"http://127.0.0.1:{self.port}/chat/completions"


class _FakeOpenAISolver:
    """Duck-types the OpenAIChatCompletionsSolver config the proxy reads."""

    def __init__(self, api_url, system_prompt="You are the persona.", timeout=None):
        self.api_url = api_url
        self.engine = "upstream-model"
        self.key = "sk-test"
        self.system_prompt = system_prompt
        self.config = {"request_timeout": timeout} if timeout else {}


def _make_app(primary_solver, supports_tools_engine=None):
    from ovos_persona_server.chat import chat_router
    from ovos_persona_server.persona import get_default_persona

    persona = MagicMock()
    persona.name = "tool-persona"
    modules = []
    if primary_solver is not None:
        modules.append(primary_solver)
    if supports_tools_engine is not None:
        modules.append(supports_tools_engine)
    persona.solvers.modules = modules

    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_default_persona] = lambda: persona
    return app, persona


def test_tools_proxied_and_tool_calls_relayed():
    with _StubUpstream() as stub:
        solver = _FakeOpenAISolver(stub.chat_url)
        app, _ = _make_app(solver)
        r = TestClient(app).post("/openai/v1/chat/completions", json={
            "model": "tool-persona",
            "messages": [{"role": "user", "content": "weather in Lisbon?"}],
            "tools": _TOOLS,
        })
    assert r.status_code == 200
    body = r.json()
    # upstream tool_calls relayed verbatim
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Lisbon"}
    # relayed body keeps the upstream model
    assert body["model"] == "upstream-model"
    # upstream actually received the tools and the injected system prompt
    sent = stub.captured[0]
    assert sent["tools"][0]["function"]["name"] == "get_weather"
    assert sent["messages"][0] == {"role": "system", "content": "You are the persona."}
    assert sent["model"] == "upstream-model"


def test_streaming_tools_deltas_relayed():
    with _StubUpstream() as stub:
        solver = _FakeOpenAISolver(stub.chat_url)
        app, _ = _make_app(solver)
        client = TestClient(app)
        with client.stream("POST", "/openai/v1/chat/completions", json={
            "model": "tool-persona",
            "messages": [{"role": "user", "content": "weather in Lisbon?"}],
            "tools": _TOOLS,
            "stream": True,
        }) as resp:
            assert resp.status_code == 200
            raw = "".join(resp.iter_text())
    assert stub.captured[0]["stream"] is True
    # tool_call deltas and the terminal marker were relayed
    assert "tool_calls" in raw
    assert '"name":"get_weather"' in raw
    assert '"finish_reason":"tool_calls"' in raw
    assert "[DONE]" in raw


def test_no_tools_uses_text_path_not_proxy(monkeypatch):
    import ovos_persona_server.chat as chat_mod
    called = {"proxy": False}

    async def _boom(*a, **k):
        called["proxy"] = True
        raise AssertionError("proxy must not run without tools")

    monkeypatch.setattr(chat_mod, "handle_tools_passthrough", _boom)
    monkeypatch.setattr(chat_mod, "run_chat", lambda *a, **k: "plain text answer")

    solver = _FakeOpenAISolver("http://127.0.0.1:1/chat/completions")
    app, _ = _make_app(solver)
    r = TestClient(app).post("/openai/v1/chat/completions", json={
        "model": "tool-persona",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert called["proxy"] is False
    assert r.json()["choices"][0]["message"]["content"] == "plain text answer"


def test_non_openai_solver_with_tools_falls_back_and_does_not_hang():
    # primary solver is NOT OpenAI (no api_url/engine/key) and no tool-capable
    # engine present -> existing 501 fallback, no crash, no hang.
    non_openai = MagicMock(spec=[])  # spec=[] -> hasattr is False for everything
    app, _ = _make_app(non_openai)
    r = TestClient(app, raise_server_exceptions=False).post("/openai/v1/chat/completions", json={
        "model": "tool-persona",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": _TOOLS,
    })
    assert r.status_code == 501


def test_upstream_error_relayed_as_openai_shaped_error():
    with _StubUpstream(mode="error") as stub:
        solver = _FakeOpenAISolver(stub.chat_url)
        app, _ = _make_app(solver)
        r = TestClient(app, raise_server_exceptions=False).post("/openai/v1/chat/completions", json={
            "model": "tool-persona",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        })
    assert r.status_code == 500
    assert r.json()["error"]["message"] == "boom"


def test_upstream_timeout_returns_error_no_hang():
    with _StubUpstream(delay=2.0) as stub:
        solver = _FakeOpenAISolver(stub.chat_url, timeout=(1, 1))
        app, _ = _make_app(solver)
        start = time.time()
        r = TestClient(app, raise_server_exceptions=False).post("/openai/v1/chat/completions", json={
            "model": "tool-persona",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        })
        elapsed = time.time() - start
    assert r.status_code == 504
    assert "timed out" in r.json()["error"]["message"]
    assert elapsed < 5  # did not hang on the 2s upstream / infinite


def test_system_prompt_not_doubled_when_caller_sends_system():
    with _StubUpstream() as stub:
        solver = _FakeOpenAISolver(stub.chat_url, system_prompt="PERSONA-DEFAULT")
        app, _ = _make_app(solver)
        TestClient(app).post("/openai/v1/chat/completions", json={
            "model": "tool-persona",
            "messages": [
                {"role": "system", "content": "CALLER-SYSTEM"},
                {"role": "user", "content": "hi"},
            ],
            "tools": _TOOLS,
        })
    sent = stub.captured[0]["messages"]
    system_msgs = [m for m in sent if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "CALLER-SYSTEM"


def test_multiturn_tool_results_passed_through_verbatim():
    with _StubUpstream() as stub:
        solver = _FakeOpenAISolver(stub.chat_url)
        app, _ = _make_app(solver)
        msgs = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_abc", "type": "function",
                             "function": {"name": "get_weather", "arguments": '{"city": "Lisbon"}'}}]},
            {"role": "tool", "content": "22C", "tool_call_id": "call_abc"},
        ]
        TestClient(app).post("/openai/v1/chat/completions", json={
            "model": "tool-persona", "messages": msgs, "tools": _TOOLS,
        })
    sent = stub.captured[0]["messages"]
    # system injected, then the three turns verbatim (assistant tool_calls + tool result)
    tool_turn = next(m for m in sent if m["role"] == "tool")
    assert tool_turn["tool_call_id"] == "call_abc" and tool_turn["content"] == "22C"
    asst = next(m for m in sent if m["role"] == "assistant")
    assert asst["tool_calls"][0]["function"]["name"] == "get_weather"
