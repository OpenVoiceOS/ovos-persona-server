"""Unit tests for honoring OpenAI ``tool_choice`` on /openai/v1/chat/completions.

The engine here is a deterministic stub that *would* call a tool whenever any
tool is offered to it (it does not decide based on ``tool_choice`` at all --
that is the server's job). This makes it possible to assert, without a real
model, that ``tool_choice="none"`` actually prevents the tool call rather than
merely being echoed back unhonored.
"""
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole, ToolCall

_TOOLS = [
    {"type": "function", "function": {
        "name": "calc", "description": "add", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "other", "description": "other", "parameters": {"type": "object", "properties": {}}}},
]


class _EagerToolEngine:
    """Calls the first offered tool whenever any tools are offered at all.

    Records the tool specs it was actually offered on each call, so tests can
    assert on what the server chose to forward rather than trusting the
    response alone.
    """
    supports_tools = True

    def __init__(self):
        self.seen_tools = None

    def continue_chat(self, messages, session_id="default", lang=None, units=None, tools=None):
        self.seen_tools = tools
        if not tools:
            return AgentMessage(role=MessageRole.ASSISTANT, content="42")
        name = tools[0]["function"]["name"]
        return AgentMessage(role=MessageRole.ASSISTANT, content="",
                            tool_calls=[ToolCall(id="c1", name=name, arguments={})])


def _make_app(engine):
    from ovos_persona_server.chat import chat_router
    from ovos_persona_server.persona import get_default_persona

    persona = MagicMock()
    persona.name = "tool-persona"
    persona.solvers.modules = [engine]

    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_default_persona] = lambda: persona
    return app


def test_tool_choice_none_prevents_tool_call():
    eng = _EagerToolEngine()
    app = _make_app(eng)
    r = TestClient(app).post(
        "/openai/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "2?"}],
              "tools": _TOOLS, "tool_choice": "none"})
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    # the engine was offered nothing, so it could not have called a tool
    assert not eng.seen_tools
    assert choice["finish_reason"] == "stop"
    assert not choice["message"]["tool_calls"]


def test_tool_choice_named_function_constrains_offer():
    eng = _EagerToolEngine()
    app = _make_app(eng)
    r = TestClient(app).post(
        "/openai/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "2?"}],
              "tools": _TOOLS,
              "tool_choice": {"type": "function", "function": {"name": "other"}}})
    assert r.status_code == 200
    # only the named function was offered to the engine, not "calc" too
    assert [t["function"]["name"] for t in eng.seen_tools] == ["other"]
    choice = r.json()["choices"][0]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "other"


def test_tool_choice_named_function_not_offered_is_rejected():
    eng = _EagerToolEngine()
    app = _make_app(eng)
    r = TestClient(app, raise_server_exceptions=False).post(
        "/openai/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "2?"}],
              "tools": _TOOLS,
              "tool_choice": {"type": "function", "function": {"name": "nonexistent"}}})
    assert r.status_code == 422
    assert eng.seen_tools is None


def test_tool_choice_required_cannot_be_forced_is_rejected():
    eng = _EagerToolEngine()
    app = _make_app(eng)
    r = TestClient(app, raise_server_exceptions=False).post(
        "/openai/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "2?"}],
              "tools": _TOOLS, "tool_choice": "tool"})
    assert r.status_code == 422
    assert eng.seen_tools is None


def test_tool_choice_auto_preserves_todays_behaviour():
    eng = _EagerToolEngine()
    app = _make_app(eng)
    r = TestClient(app).post(
        "/openai/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "2?"}],
              "tools": _TOOLS, "tool_choice": "auto"})
    assert r.status_code == 200
    assert [t["function"]["name"] for t in eng.seen_tools] == ["calc", "other"]
    choice = r.json()["choices"][0]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "calc"


def test_tool_choice_absent_preserves_todays_behaviour():
    eng = _EagerToolEngine()
    app = _make_app(eng)
    r = TestClient(app).post(
        "/openai/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "2?"}],
              "tools": _TOOLS})
    assert r.status_code == 200
    assert [t["function"]["name"] for t in eng.seen_tools] == ["calc", "other"]
    choice = r.json()["choices"][0]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "calc"
