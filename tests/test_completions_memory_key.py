"""The legacy completions endpoint must key memory on the caller's ``user``.

``CreateCompletionRequest`` accepts the OpenAI ``user`` field exactly like the
chat endpoint does, and ``CHAT_MEMORY=transparent`` is documented to key history
by that field. These tests pin that the field is not discarded on the way in.
"""
import json
from typing import Dict, Generator, List

import pytest
from fastapi.testclient import TestClient

import ovos_persona_server
from ovos_persona_server import create_persona_app
from ovos_persona_server import persona as persona_mod


class _RecordingMemory:
    """Memory plugin double that records which session key it was asked for."""

    def __init__(self) -> None:
        self.history: Dict[str, list] = {}

    def build_conversation_context(self, utterance, session_id):
        self.history.setdefault(session_id, [])
        return list(self.history[session_id]) + [utterance]

    def update_history(self, new_messages, session_id):
        self.history.setdefault(session_id, []).extend(new_messages)


class _FakePersona:
    """Minimal stand-in for ovos_persona.Persona."""

    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.memory = None

        class _Solvers:
            loaded_modules: dict = {}

        self.solvers = _Solvers()

    def chat(self, messages, sess=None, **kwargs) -> str:
        return "ok"

    def stream(self, messages, sess=None, **kwargs) -> Generator[str, None, None]:
        yield "ok"


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(persona_mod.personas)
    saved_default = persona_mod.default_persona
    yield
    persona_mod.personas.clear()
    persona_mod.personas.update(saved)
    persona_mod.default_persona = saved_default


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setattr(ovos_persona_server, "Persona", _FakePersona)
    monkeypatch.setattr(persona_mod.settings, "chat_memory", "transparent")
    path = tmp_path / "alpha.json"
    path.write_text(json.dumps({"name": "alpha", "solvers": ["fake"]}))
    app = create_persona_app(str(path))
    persona_mod.personas["alpha"].memory = _RecordingMemory()
    return TestClient(app)


def _memory() -> _RecordingMemory:
    return persona_mod.personas["alpha"].memory


def _complete(client: TestClient, prompt: str, user: str) -> None:
    resp = client.post("/openai/v1/completions",
                       json={"prompt": prompt, "user": user})
    assert resp.status_code == 200


def test_distinct_users_do_not_share_history(client):
    _complete(client, "the code is hunter2", user="caller-a")
    _complete(client, "what is the code", user="caller-b")

    assert set(_memory().history) == {"caller-a", "caller-b"}
    assert "hunter2" not in str(_memory().history["caller-b"])


def test_same_user_keeps_its_own_history(client):
    _complete(client, "the code is hunter2", user="caller-a")
    _complete(client, "what is the code", user="caller-a")

    assert list(_memory().history) == ["caller-a"]
    assert "hunter2" in str(_memory().history["caller-a"])


def test_streaming_completions_key_on_user(client):
    with client.stream("POST", "/openai/v1/completions",
                       json={"prompt": "the code is hunter2", "stream": True,
                             "user": "caller-a"}) as resp:
        assert resp.status_code == 200
        list(resp.iter_lines())
    with client.stream("POST", "/openai/v1/completions",
                       json={"prompt": "what is the code", "stream": True,
                             "user": "caller-b"}) as resp:
        assert resp.status_code == 200
        list(resp.iter_lines())

    assert set(_memory().history) == {"caller-a", "caller-b"}
    assert "hunter2" not in str(_memory().history["caller-b"])
