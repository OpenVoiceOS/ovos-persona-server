"""Tests for the client system-prompt strategies (ignore | replace | append)."""
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ovos_persona_server.system_prompt import (
    apply_system_prompt_strategy, resolve_strategy, DEFAULT_STRATEGY,
)

PERSONA_SP = "PERSONA-IDENTITY"
CLIENT_SP = "CLIENT-CONTEXT"


def _persona(strategy=None, system_prompt=PERSONA_SP):
    solver = types.SimpleNamespace(system_prompt=system_prompt)
    persona = types.SimpleNamespace()
    persona.name = "p"
    persona.config = {"system_prompt_strategy": strategy} if strategy else {}
    persona.solvers = types.SimpleNamespace(modules=[solver], loaded_modules={})
    return persona


def _systems(messages):
    return [m["content"] for m in messages if m["role"] == "system"]


# --- pure strategy function -------------------------------------------------

def test_default_strategy_is_ignore():
    assert DEFAULT_STRATEGY == "ignore"
    assert resolve_strategy(_persona()) == "ignore"


def test_ignore_drops_client_system():
    msgs = [{"role": "system", "content": CLIENT_SP}, {"role": "user", "content": "hi"}]
    out = apply_system_prompt_strategy(_persona("ignore"), msgs)
    assert _systems(out) == [PERSONA_SP]
    assert out[-1] == {"role": "user", "content": "hi"}


def test_replace_uses_client_system():
    msgs = [{"role": "system", "content": CLIENT_SP}, {"role": "user", "content": "hi"}]
    out = apply_system_prompt_strategy(_persona("replace"), msgs)
    assert _systems(out) == [CLIENT_SP]


def test_replace_falls_back_to_persona_when_no_client_system():
    msgs = [{"role": "user", "content": "hi"}]
    out = apply_system_prompt_strategy(_persona("replace"), msgs)
    assert _systems(out) == [PERSONA_SP]


def test_append_combines_persona_then_client():
    msgs = [{"role": "system", "content": CLIENT_SP}, {"role": "user", "content": "hi"}]
    out = apply_system_prompt_strategy(_persona("append"), msgs)
    assert _systems(out) == [f"{PERSONA_SP}\n\n{CLIENT_SP}"]


def test_append_without_client_system_is_just_persona():
    msgs = [{"role": "user", "content": "hi"}]
    out = apply_system_prompt_strategy(_persona("append"), msgs)
    assert _systems(out) == [PERSONA_SP]


def test_multiple_client_systems_concatenated():
    msgs = [
        {"role": "system", "content": "A"},
        {"role": "system", "content": "B"},
        {"role": "user", "content": "hi"},
    ]
    out = apply_system_prompt_strategy(_persona("replace"), msgs)
    assert _systems(out) == ["A\n\nB"]


def test_ignore_without_persona_prompt_yields_no_system():
    # no persona system prompt -> nothing injected, downstream solver decides
    out = apply_system_prompt_strategy(_persona("ignore", system_prompt=None),
                                       [{"role": "system", "content": CLIENT_SP},
                                        {"role": "user", "content": "hi"}])
    assert _systems(out) == []


def test_env_var_used_when_config_absent(monkeypatch):
    monkeypatch.setenv("PERSONA_SYSTEM_PROMPT_STRATEGY", "append")
    assert resolve_strategy(_persona()) == "append"


def test_unknown_strategy_falls_back_to_ignore():
    assert resolve_strategy(_persona("nonsense")) == "ignore"


# --- end-to-end through the text path (Persona.chat) ------------------------

class _RecordingPersona:
    """Captures the AgentMessages that reach Persona.chat."""

    def __init__(self, strategy=None):
        self.name = "rec"
        self.config = {"system_prompt_strategy": strategy} if strategy else {}
        self.memory = None
        self.solvers = types.SimpleNamespace(
            modules=[types.SimpleNamespace(system_prompt=PERSONA_SP)], loaded_modules={})
        self.captured = None

    def chat(self, messages, sess=None):
        self.captured = messages
        return "ok"

    def stream(self, messages, sess=None):
        self.captured = messages
        yield "ok"


def _app(persona):
    from ovos_persona_server.chat import chat_router
    from ovos_persona_server.persona import get_default_persona
    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_default_persona] = lambda: persona
    return TestClient(app)


def _captured_systems(persona):
    return [m.content for m in persona.captured
            if getattr(m.role, "value", m.role) == "system"]


def test_text_path_ignore(monkeypatch):
    monkeypatch.delenv("PERSONA_SYSTEM_PROMPT_STRATEGY", raising=False)
    persona = _RecordingPersona("ignore")
    r = _app(persona).post("/openai/v1/chat/completions", json={
        "model": "rec",
        "messages": [{"role": "system", "content": CLIENT_SP}, {"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert _captured_systems(persona) == [PERSONA_SP]


def test_text_path_replace(monkeypatch):
    monkeypatch.delenv("PERSONA_SYSTEM_PROMPT_STRATEGY", raising=False)
    persona = _RecordingPersona("replace")
    _app(persona).post("/openai/v1/chat/completions", json={
        "model": "rec",
        "messages": [{"role": "system", "content": CLIENT_SP}, {"role": "user", "content": "hi"}],
    })
    assert _captured_systems(persona) == [CLIENT_SP]


def test_text_path_append(monkeypatch):
    monkeypatch.delenv("PERSONA_SYSTEM_PROMPT_STRATEGY", raising=False)
    persona = _RecordingPersona("append")
    _app(persona).post("/openai/v1/chat/completions", json={
        "model": "rec",
        "messages": [{"role": "system", "content": CLIENT_SP}, {"role": "user", "content": "hi"}],
    })
    assert _captured_systems(persona) == [f"{PERSONA_SP}\n\n{CLIENT_SP}"]


def test_text_path_default_ignores_client_system(monkeypatch):
    # no strategy configured anywhere -> default ignore
    monkeypatch.delenv("PERSONA_SYSTEM_PROMPT_STRATEGY", raising=False)
    persona = _RecordingPersona()
    _app(persona).post("/openai/v1/chat/completions", json={
        "model": "rec",
        "messages": [{"role": "system", "content": CLIENT_SP}, {"role": "user", "content": "hi"}],
    })
    assert _captured_systems(persona) == [PERSONA_SP]
