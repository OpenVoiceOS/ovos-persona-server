"""Unit tests for tool/function calling on /openai/v1/chat/completions."""
import json
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole, ToolCall

_TOOLS = [{"type": "function", "function": {
    "name": "calc", "description": "add", "parameters": {"type": "object", "properties": {}}}}]


class _ToolEngine:
    """Minimal tool-capable chat handler recording what it received."""
    supports_tools = True

    def __init__(self, resp):
        self._resp = resp
        self.seen_tools = None
        self.seen_messages = None

    def continue_chat(self, messages, session_id="default", lang=None, units=None, tools=None):
        self.seen_tools = tools
        self.seen_messages = messages
        return self._resp


def _make_app(engine):
    from ovos_persona_server.chat import chat_router
    from ovos_persona_server.persona import get_default_persona

    persona = MagicMock()
    persona.name = "tool-persona"
    persona.solvers.modules = [engine] if engine is not None else []

    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_default_persona] = lambda: persona
    return app, engine


def test_tools_return_tool_calls():
    eng = _ToolEngine(AgentMessage(role=MessageRole.ASSISTANT, content="",
                                   tool_calls=[ToolCall(id="c1", name="calc", arguments={"a": 1})]))
    app, _ = _make_app(eng)
    client = TestClient(app)
    r = client.post("/openai/v1/chat/completions",
                    json={"model": "x", "messages": [{"role": "user", "content": "2?"}], "tools": _TOOLS})
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "calc"
    assert json.loads(tc["function"]["arguments"]) == {"a": 1}
    # the OpenAI tool specs were forwarded to the engine
    assert eng.seen_tools[0]["function"]["name"] == "calc"


def test_plain_answer_when_model_does_not_call():
    eng = _ToolEngine(AgentMessage(role=MessageRole.ASSISTANT, content="42"))
    app, _ = _make_app(eng)
    r = TestClient(app).post("/openai/v1/chat/completions",
                             json={"model": "x", "messages": [{"role": "user", "content": "6*7?"}], "tools": _TOOLS})
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "42"
    assert not choice["message"]["tool_calls"]


def test_tools_without_capable_engine_returns_501():
    app, _ = _make_app(None)
    r = TestClient(app, raise_server_exceptions=False).post(
        "/openai/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "tools": _TOOLS})
    assert r.status_code == 501


def test_streaming_with_client_tool_call_streams_tool_calls_delta():
    eng = _ToolEngine(AgentMessage(role=MessageRole.ASSISTANT, content="",
                                   tool_calls=[ToolCall(id="c1", name="calc", arguments={"a": 1})]))
    app, _ = _make_app(eng)
    with TestClient(app).stream(
            "POST", "/openai/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "2?"}],
                  "tools": _TOOLS, "stream": True}) as resp:
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())
    assert '"name":"calc"' in raw or '"name": "calc"' in raw
    assert "tool_calls" in raw
    assert "[DONE]" in raw


def test_streaming_with_tools_plain_answer_streams_content():
    eng = _ToolEngine(AgentMessage(role=MessageRole.ASSISTANT, content="42"))
    app, _ = _make_app(eng)
    with TestClient(app).stream(
            "POST", "/openai/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "6*7?"}],
                  "tools": _TOOLS, "stream": True}) as resp:
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())
    assert "42" in raw
    assert '"finish_reason":"stop"' in raw or '"finish_reason": "stop"' in raw
    assert "[DONE]" in raw


def test_incoming_tool_messages_converted_for_engine():
    eng = _ToolEngine(AgentMessage(role=MessageRole.ASSISTANT, content="final"))
    app, _ = _make_app(eng)
    msgs = [
        {"role": "user", "content": "2+3?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "calc", "arguments": '{"a": 2, "b": 3}'}}]},
        {"role": "tool", "content": "5", "tool_call_id": "c1"},
    ]
    r = TestClient(app).post("/openai/v1/chat/completions",
                             json={"model": "x", "messages": msgs, "tools": _TOOLS})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "final"
    # the assistant tool_calls turn and the tool result were rebuilt as AgentMessages
    tool_msg = next(m for m in eng.seen_messages if m.role == MessageRole.TOOL)
    assert tool_msg.tool_call_id == "c1" and tool_msg.content == "5"
    asst = next(m for m in eng.seen_messages if m.role == MessageRole.ASSISTANT and m.tool_calls)
    assert asst.tool_calls[0].name == "calc" and asst.tool_calls[0].arguments == {"a": 2, "b": 3}
