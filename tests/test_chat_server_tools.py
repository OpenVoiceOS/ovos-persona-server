"""Tests for persona-owned (server-side) ToolBox participation in chat.

These exercise the OVOS abstraction end to end: a tool-capable engine
(``supports_tools``) is offered the persona's own ToolBox tools alongside any
client tools; calls to a ToolBox tool are executed server-side in an agentic
loop and fed back to the model, while calls to a client tool are relayed to the
caller. No network is used — a fake engine scripts the tool_calls and a fake
ToolBox registry stands in for installed plugins.
"""
import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole, ToolCall
import ovos_persona_server.chat as chat_mod
import ovos_persona_server.server_tools as server_tools

_CLIENT_TOOLS = [{"type": "function", "function": {
    "name": "client_lookup", "description": "client executes this",
    "parameters": {"type": "object", "properties": {}}}}]


class _ScriptedEngine:
    """Tool-capable engine that returns a queued response per continue_chat call."""
    supports_tools = True

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []  # (messages, tools) seen each turn

    def continue_chat(self, messages, session_id="default", lang=None, units=None, tools=None):
        self.calls.append((list(messages), tools))
        return self._responses.pop(0)


def _fake_tool(name, schema=None):
    tool = MagicMock()
    tool.description = f"{name} description"
    tool.argument_schema.model_json_schema.return_value = schema or {
        "type": "object", "properties": {"x": {"type": "string"}}}
    return tool


def _fake_registry(results):
    """Build a {name: (toolbox, tool)} registry executing from ``results`` map."""
    registry = {}
    for name in results:
        toolbox = MagicMock()

        def _call(short, kwargs, _n=name):
            out = MagicMock()
            out.model_dump.return_value = results[_n]
            return out

        toolbox.call_tool.side_effect = _call
        registry[name] = (toolbox, _fake_tool(name))
    return registry


@pytest.fixture(autouse=True)
def _clear_registry_cache(monkeypatch):
    server_tools.reset_registry_cache()
    yield
    server_tools.reset_registry_cache()


def _make_app(engine, registry, monkeypatch):
    from ovos_persona_server.persona import get_default_persona

    persona = MagicMock()
    persona.name = "tool-persona"
    persona.solvers.modules = [engine] if engine is not None else []

    # chat.py resolves the persona's ToolBox registry through this seam.
    monkeypatch.setattr(chat_mod, "cached_registry", lambda: registry)

    app = FastAPI()
    app.include_router(chat_mod.chat_router)
    app.dependency_overrides[get_default_persona] = lambda: persona
    return app, persona


def test_persona_toolbox_tool_executed_server_side(monkeypatch):
    # turn 1: model calls the server tool; turn 2: model answers with the result.
    eng = _ScriptedEngine([
        AgentMessage(role=MessageRole.ASSISTANT, content="",
                     tool_calls=[ToolCall(id="s1", name="weather", arguments={"city": "Lisbon"})]),
        AgentMessage(role=MessageRole.ASSISTANT, content="It is 22C in Lisbon."),
    ])
    registry = _fake_registry({"weather": {"temp": "22C"}})
    app, _ = _make_app(eng, registry, monkeypatch)

    r = TestClient(app).post("/openai/v1/chat/completions",
                             json={"model": "x", "messages": [{"role": "user", "content": "weather?"}]})
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    # server ran the tool and the model produced a final answer -> plain stop
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "It is 22C in Lisbon."
    # the ToolBox actually executed
    registry["weather"][0].call_tool.assert_called_once()
    # the tool was offered to the engine on the first turn
    _msgs, tools = eng.calls[0]
    assert any(t["function"]["name"] == "weather" for t in tools)
    # second turn saw the assistant tool_calls turn + the tool result message
    msgs2, _ = eng.calls[1]
    assert any(m.role == MessageRole.TOOL and m.tool_call_id == "s1" for m in msgs2)


def test_client_tool_call_relayed_not_executed(monkeypatch):
    eng = _ScriptedEngine([
        AgentMessage(role=MessageRole.ASSISTANT, content="",
                     tool_calls=[ToolCall(id="c1", name="client_lookup", arguments={"q": "x"})]),
    ])
    registry = _fake_registry({})  # no server tools
    app, _ = _make_app(eng, registry, monkeypatch)

    r = TestClient(app).post("/openai/v1/chat/completions",
                             json={"model": "x", "messages": [{"role": "user", "content": "hi"}],
                                   "tools": _CLIENT_TOOLS})
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "client_lookup"
    # only one engine turn: client tool is relayed, never looped
    assert len(eng.calls) == 1


def test_client_and_server_tools_offered_together(monkeypatch):
    eng = _ScriptedEngine([AgentMessage(role=MessageRole.ASSISTANT, content="done")])
    registry = _fake_registry({"weather": {"temp": "22C"}})
    app, _ = _make_app(eng, registry, monkeypatch)

    TestClient(app).post("/openai/v1/chat/completions",
                         json={"model": "x", "messages": [{"role": "user", "content": "hi"}],
                               "tools": _CLIENT_TOOLS})
    _msgs, tools = eng.calls[0]
    names = {t["function"]["name"] for t in tools}
    assert {"client_lookup", "weather"} <= names


def test_server_tool_loop_bounded(monkeypatch):
    # engine always calls the server tool -> loop must stop at MAX_TOOL_ITERS.
    always_call = [
        AgentMessage(role=MessageRole.ASSISTANT, content="",
                     tool_calls=[ToolCall(id=f"s{i}", name="weather", arguments={})])
        for i in range(server_tools.MAX_TOOL_ITERS + 3)
    ]
    eng = _ScriptedEngine(always_call)
    registry = _fake_registry({"weather": {"temp": "22C"}})
    app, _ = _make_app(eng, registry, monkeypatch)

    r = TestClient(app).post("/openai/v1/chat/completions",
                             json={"model": "x", "messages": [{"role": "user", "content": "loop"}]})
    assert r.status_code == 200
    assert len(eng.calls) == server_tools.MAX_TOOL_ITERS


def test_no_tools_and_no_toolboxes_uses_text_path(monkeypatch):
    # engine present but no client tools and empty registry -> plain text path.
    eng = _ScriptedEngine([AgentMessage(role=MessageRole.ASSISTANT, content="unused")])
    registry = _fake_registry({})
    app, _ = _make_app(eng, registry, monkeypatch)
    monkeypatch.setattr(chat_mod, "run_chat", lambda *a, **k: "plain text")

    r = TestClient(app).post("/openai/v1/chat/completions",
                             json={"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "plain text"
    # tool loop never ran
    assert eng.calls == []


def test_streaming_executes_server_tool_then_streams_answer(monkeypatch):
    eng = _ScriptedEngine([
        AgentMessage(role=MessageRole.ASSISTANT, content="",
                     tool_calls=[ToolCall(id="s1", name="weather", arguments={"city": "Lisbon"})]),
        AgentMessage(role=MessageRole.ASSISTANT, content="It is 22C in Lisbon."),
    ])
    registry = _fake_registry({"weather": {"temp": "22C"}})
    app, _ = _make_app(eng, registry, monkeypatch)

    with TestClient(app).stream(
            "POST", "/openai/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "weather?"}],
                  "stream": True}) as resp:
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())
    # the server tool ran, and the final answer was streamed
    registry["weather"][0].call_tool.assert_called_once()
    assert "22C in Lisbon" in raw
    assert "[DONE]" in raw


def test_server_tool_failure_surfaced_to_model(monkeypatch):
    reg = {}
    toolbox = MagicMock()
    toolbox.call_tool.side_effect = RuntimeError("boom")
    reg["weather"] = (toolbox, _fake_tool("weather"))
    eng = _ScriptedEngine([
        AgentMessage(role=MessageRole.ASSISTANT, content="",
                     tool_calls=[ToolCall(id="s1", name="weather", arguments={})]),
        AgentMessage(role=MessageRole.ASSISTANT, content="sorry, tool failed"),
    ])
    app, _ = _make_app(eng, reg, monkeypatch)

    r = TestClient(app).post("/openai/v1/chat/completions",
                             json={"model": "x", "messages": [{"role": "user", "content": "weather?"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "sorry, tool failed"
    # the error was fed back to the model as a tool message
    msgs2, _ = eng.calls[1]
    tool_msg = next(m for m in msgs2 if m.role == MessageRole.TOOL)
    assert "boom" in json.loads(tool_msg.content)["error"]
