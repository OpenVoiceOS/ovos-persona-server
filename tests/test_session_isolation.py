"""Callers of the HTTP routers must not share a solver conversation.

A stateful solver (a remote A2A agent, a vendor chat thread) keys its state by
``sess.session_id``. A caller that identifies itself keeps one stable session
that no other caller reaches. Callers that send no identity resolve to the
reserved ``"default"`` session and keep their own follow-ups.

OVOS-SESSION-1 §3.1: two Messages with distinct ``session_id`` values do not
belong to the same session, and a consumer that maintains per-session state
MUST key that state on ``session_id``. §2.1: an omitted ``session_id`` is
filled by the consumer with the reserved value ``"default"``.
"""
import json
from typing import Dict, Generator, List
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

import ovos_persona_server
import ovos_persona_server.chat as chat_mod
import ovos_persona_server.server_tools as server_tools
from ovos_persona_server import create_persona_app
from ovos_persona_server import persona as persona_mod
from tests.test_chat_server_tools import _fake_registry


class _Memory:
    """In-test memory plugin with the persona memory interface."""

    def __init__(self) -> None:
        self.history: Dict[str, list] = {}

    def build_conversation_context(self, utterance, session_id):
        return list(self.history.get(session_id, [])) + [AgentMessage(MessageRole.USER, utterance)]

    def update_history(self, new_messages, session_id):
        self.history.setdefault(session_id, []).extend(new_messages)


class _Backend:
    """Stateful backend: remembers one secret per session_id."""

    def __init__(self) -> None:
        self.secrets: Dict[str, str] = {}
        self.replies: List[str] = []

    def answer(self, text: str, session_id: str) -> str:
        if text.startswith("my secret is "):
            self.secrets[session_id] = text[len("my secret is "):]
            reply = "noted"
        else:
            reply = self.secrets.get(session_id, "unknown")
        self.replies.append(reply)
        return reply


class _StatefulPersona:
    """Persona double that routes chat and stream to the stateful backend."""

    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.memory = None
        self.backend = _Backend()

        class _Solvers:
            loaded_modules: dict = {}

        self.solvers = _Solvers()

    def chat(self, messages, sess=None, **kwargs) -> str:
        return self.backend.answer(messages[-1].content, sess.session_id)

    def stream(self, messages, sess=None, **kwargs) -> Generator[str, None, None]:
        yield self.backend.answer(messages[-1].content, sess.session_id)


@pytest.fixture(params=["off", "transparent"])
def client(request, tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setattr(ovos_persona_server, "Persona", _StatefulPersona)
    monkeypatch.setattr(persona_mod.settings, "chat_memory", request.param)
    saved = dict(persona_mod.personas)
    saved_default = persona_mod.default_persona
    path = tmp_path / "alpha.json"
    path.write_text(json.dumps({"name": "alpha", "solvers": ["fake"]}))
    app = create_persona_app(str(path))
    if request.param == "transparent":
        persona_mod.personas["alpha"].memory = _Memory()
    yield TestClient(app)
    persona_mod.personas.clear()
    persona_mod.personas.update(saved)
    persona_mod.default_persona = saved_default


def _replies() -> List[str]:
    return persona_mod.personas["alpha"].backend.replies


def _chat(client, text, user=None, stream=False):
    body = {"model": "alpha", "stream": stream,
            "messages": [{"role": "user", "content": text}]}
    if user:
        body["user"] = user
    if stream:
        with client.stream("POST", "/openai/v1/chat/completions", json=body) as resp:
            assert resp.status_code == 200
            list(resp.iter_lines())
    else:
        assert client.post("/openai/v1/chat/completions", json=body).status_code == 200
    return _replies()[-1]


def _anthropic(client, text, stream=False):
    resp = client.post("/anthropic/v1/messages", json={
        "model": "alpha", "max_tokens": 64, "stream": stream,
        "messages": [{"role": "user", "content": text}]})
    assert resp.status_code == 200
    return _replies()[-1]


def _ollama(client, text, stream=False):
    resp = client.post("/ollama/api/chat", json={
        "model": "alpha", "stream": stream,
        "messages": [{"role": "user", "content": text}]})
    assert resp.status_code == 200
    return _replies()[-1]


@pytest.mark.parametrize("stream", [False, True])
def test_identified_callers_get_separate_solver_sessions(client, stream):
    assert _chat(client, "my secret is 4471", user="alice", stream=stream) == "noted"

    assert _chat(client, "what is my secret", user="bob", stream=stream) == "unknown"
    assert _chat(client, "what is my secret", user="alice", stream=stream) == "4471"


@pytest.mark.parametrize("stream", [False, True])
def test_anonymous_and_identified_callers_do_not_reach_each_other(client, stream):
    _chat(client, "my secret is 4471", user="alice", stream=stream)
    assert _chat(client, "what is my secret", stream=stream) == "unknown"

    _chat(client, "my secret is 9", stream=stream)
    assert _chat(client, "what is my secret", user="alice", stream=stream) == "4471"
    assert _chat(client, "what is my secret", user="bob", stream=stream) == "unknown"


@pytest.mark.parametrize("stream", [False, True])
def test_anonymous_follow_up_keeps_its_context(client, stream):
    assert _chat(client, "my secret is 9", stream=stream) == "noted"
    assert _chat(client, "what is my secret", stream=stream) == "9"


def _ollama_generate(client, text, stream=False):
    resp = client.post("/ollama/api/generate", json={
        "model": "alpha", "stream": stream, "prompt": text})
    assert resp.status_code == 200
    return _replies()[-1]


@pytest.mark.parametrize("send", [_anthropic, _ollama, _ollama_generate])
@pytest.mark.parametrize("stream", [False, True])
def test_router_without_a_user_field_keeps_context_and_isolates_identified(client, send, stream):
    _chat(client, "my secret is 4471", user="alice")

    assert send(client, "what is my secret", stream=stream) == "unknown"
    send(client, "my secret is 9", stream=stream)
    assert send(client, "what is my secret", stream=stream) == "9"
    assert _chat(client, "what is my secret", user="alice") == "4471"


# --- tool-capable chat branch -------------------------------------------------

class _ToolEngine:
    """Tool-capable chat engine over the stateful backend."""
    supports_tools = True

    def __init__(self, backend: _Backend) -> None:
        self.backend = backend

    def continue_chat(self, messages, session_id="default", **kwargs):
        text = self.backend.answer(messages[-1].content, session_id)
        return AgentMessage(role=MessageRole.ASSISTANT, content=text)


@pytest.fixture()
def tool_client(monkeypatch):
    server_tools.reset_registry_cache()
    backend = _Backend()
    persona = MagicMock()
    persona.name = "tool-persona"
    persona.solvers.modules = [_ToolEngine(backend)]
    persona.memory = None
    monkeypatch.setattr(chat_mod, "cached_registry", lambda: _fake_registry({"weather": {"temp": 21}}))
    monkeypatch.setattr(persona_mod.settings, "chat_memory", "off")
    app = FastAPI()
    app.include_router(chat_mod.chat_router)
    app.dependency_overrides[persona_mod.get_default_persona] = lambda: persona
    yield TestClient(app), backend
    server_tools.reset_registry_cache()


def _tool_chat(client, text, user=None):
    body = {"messages": [{"role": "user", "content": text}]}
    if user:
        body["user"] = user
    assert client.post("/openai/v1/chat/completions", json=body).status_code == 200


def test_tool_branch_isolates_identified_callers_and_keeps_anonymous_context(tool_client):
    client, backend = tool_client
    _tool_chat(client, "my secret is 4471", user="alice")
    _tool_chat(client, "what is my secret", user="bob")
    assert backend.replies[-1] == "unknown"
    _tool_chat(client, "what is my secret", user="alice")
    assert backend.replies[-1] == "4471"

    _tool_chat(client, "what is my secret")
    assert backend.replies[-1] == "unknown"
    _tool_chat(client, "my secret is 9")
    _tool_chat(client, "what is my secret")
    assert backend.replies[-1] == "9"
