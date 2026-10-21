"""Transparent memory must survive the tool-capable engine branch of /chat/completions.

With ``CHAT_MEMORY=transparent`` the server owns conversation state keyed by the
OpenAI ``user`` field. A persona whose engine ``supports_tools`` and which has
ToolBox plugins installed is dispatched through the server tool loop; that path
must build the turn from the persona's memory and persist the exchange exactly
like the plain ``run_chat`` path does.
"""
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole, ToolCall
import ovos_persona_server.chat as chat_mod
import ovos_persona_server.server_tools as server_tools
from ovos_persona_server import persona as persona_mod

from tests.test_chat_server_tools import _ScriptedEngine, _fake_registry


class _RecordingMemory:
    def __init__(self):
        self.history = {}

    def build_conversation_context(self, utterance, session_id):
        past = self.history.setdefault(session_id, [])
        return list(past) + [AgentMessage(role=MessageRole.USER, content=utterance)]

    def update_history(self, new_messages, session_id):
        self.history.setdefault(session_id, []).extend(new_messages)


@pytest.fixture(autouse=True)
def _registry_cache():
    server_tools.reset_registry_cache()
    yield
    server_tools.reset_registry_cache()


def _app(engine, monkeypatch, memory):
    from ovos_persona_server.persona import get_default_persona
    persona = MagicMock()
    persona.name = "tool-persona"
    persona.solvers.modules = [engine]
    persona.memory = memory
    monkeypatch.setattr(chat_mod, "cached_registry", lambda: _fake_registry({"weather": {"temp": 21}}))
    monkeypatch.setattr(persona_mod.settings, "chat_memory", "transparent")
    app = FastAPI()
    app.include_router(chat_mod.chat_router)
    app.dependency_overrides[get_default_persona] = lambda: persona
    return TestClient(app)


def _chat(client, text, user):
    r = client.post("/openai/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": text}], "user": user})
    assert r.status_code == 200, r.text
    return r.json()


def test_tool_branch_persists_and_replays_memory(monkeypatch):
    eng = _ScriptedEngine([
        AgentMessage(role=MessageRole.ASSISTANT, content="Hello Quillanova."),
        AgentMessage(role=MessageRole.ASSISTANT, content="Quillanova"),
    ])
    memory = _RecordingMemory()
    client = _app(eng, monkeypatch, memory)

    _chat(client, "My name is Quillanova.", user="alice")
    # A single registered persona keys memory by the bare user id (memory_session_id).
    sid = "alice"
    assert [m.content for m in memory.history[sid]] == ["My name is Quillanova.", "Hello Quillanova."]

    _chat(client, "What is my name?", user="alice")
    second_turn_input = [m.content for m in eng.calls[1][0]]
    assert second_turn_input == ["My name is Quillanova.", "Hello Quillanova.", "What is my name?"]
    assert [m.content for m in memory.history[sid]][-2:] == ["What is my name?", "Quillanova"]


def test_tool_branch_keys_memory_per_user(monkeypatch):
    eng = _ScriptedEngine([
        AgentMessage(role=MessageRole.ASSISTANT, content="a"),
        AgentMessage(role=MessageRole.ASSISTANT, content="b"),
    ])
    memory = _RecordingMemory()
    client = _app(eng, monkeypatch, memory)
    _chat(client, "one", user="alice")
    _chat(client, "two", user="bob")
    assert sorted(memory.history) == ["alice", "bob"]
    assert [m.content for m in eng.calls[1][0]] == ["two"]


def test_tool_branch_relayed_client_call_is_not_persisted(monkeypatch):
    """A turn that ends in a client-side tool_call is unfinished: nothing to persist yet."""
    eng = _ScriptedEngine([
        AgentMessage(role=MessageRole.ASSISTANT, content="",
                     tool_calls=[ToolCall(id="c1", name="client_lookup", arguments={})]),
    ])
    memory = _RecordingMemory()
    client = _app(eng, monkeypatch, memory)
    body = _chat_with_tools(client, "look it up", user="alice")
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert memory.history.get("alice", []) == []


def _chat_with_tools(client, text, user):
    tools = [{"type": "function", "function": {"name": "client_lookup", "description": "x",
                                                "parameters": {"type": "object", "properties": {}}}}]
    r = client.post("/openai/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": text}], "user": user, "tools": tools})
    assert r.status_code == 200, r.text
    return r.json()
