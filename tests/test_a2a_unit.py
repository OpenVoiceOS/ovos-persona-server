"""
Unit tests for the A2A adapter (ovos_persona_server/a2a.py).

All A2A SDK internals are mocked — no live server, no network needed.
"""

import asyncio
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake A2A types (mirrors real SDK field names)
# ---------------------------------------------------------------------------

class _FakeTextPart:
    def __init__(self, text: str = "", **kwargs: Any) -> None:
        self.text = text


class _FakePart:
    def __init__(self, root: Any = None, **kwargs: Any) -> None:
        self.root = root


class _FakeMessage:
    def __init__(self, text: str = "", parts: List[Any] = None, **kwargs: Any) -> None:
        if parts is not None:
            self.parts = parts
        else:
            self.parts = [_FakePart(root=_FakeTextPart(text))]


class _FakeAgentCard:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _FakeAgentCapabilities:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _FakeAgentSkill:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _FakeArtifact:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _FakeTaskArtifactUpdateEvent:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _FakeTaskStatusUpdateEvent:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _FakeTaskState:
    completed = "completed"
    canceled = "canceled"


class _FakeTaskStatus:
    def __init__(self, state: str = "", **kwargs: Any) -> None:
        self.state = state


class _FakeRequestContext:
    def __init__(self, message: Any, task_id: str = "t1", context_id: str = "c1") -> None:
        self.message = message
        self.task_id = task_id
        self.context_id = context_id


class _FakeEventQueue:
    def __init__(self) -> None:
        self.events: List[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _patch_a2a(monkeypatch):
    """Patch a2a SDK symbols in the a2a module."""
    import ovos_persona_server.a2a as a2a_mod

    monkeypatch.setattr(a2a_mod, "_A2A_AVAILABLE", True)
    monkeypatch.setattr(a2a_mod, "AgentCard", _FakeAgentCard)
    monkeypatch.setattr(a2a_mod, "AgentCapabilities", _FakeAgentCapabilities)
    monkeypatch.setattr(a2a_mod, "AgentSkill", _FakeAgentSkill)
    monkeypatch.setattr(a2a_mod, "Artifact", _FakeArtifact)
    monkeypatch.setattr(a2a_mod, "Part", _FakePart)
    monkeypatch.setattr(a2a_mod, "TextPart", _FakeTextPart)
    monkeypatch.setattr(a2a_mod, "TaskArtifactUpdateEvent", _FakeTaskArtifactUpdateEvent)
    monkeypatch.setattr(a2a_mod, "TaskStatusUpdateEvent", _FakeTaskStatusUpdateEvent)
    monkeypatch.setattr(a2a_mod, "TaskState", _FakeTaskState)
    monkeypatch.setattr(a2a_mod, "TaskStatus", _FakeTaskStatus)

    yield a2a_mod


def _fake_persona(sentences: List[str]) -> MagicMock:
    """Return a mock Persona whose stream() returns *sentences*."""
    persona = MagicMock()
    persona.name = "test-persona"
    persona.config = {}
    persona.stream.return_value = iter(sentences)
    return persona


# ---------------------------------------------------------------------------
# _extract_user_text
# ---------------------------------------------------------------------------

def test_extract_user_text_single_part(_patch_a2a) -> None:
    a2a_mod = _patch_a2a
    from ovos_persona_server.a2a import OVOSPersonaAgentExecutor

    msg = _FakeMessage("hello world")
    result = OVOSPersonaAgentExecutor._extract_user_text(msg)
    assert result == "hello world"


def test_extract_user_text_multi_part(_patch_a2a) -> None:
    from ovos_persona_server.a2a import OVOSPersonaAgentExecutor

    msg = _FakeMessage()
    msg.parts = [
        _FakePart(root=_FakeTextPart("hello")),
        _FakePart(root=_FakeTextPart("world")),
    ]
    result = OVOSPersonaAgentExecutor._extract_user_text(msg)
    assert result == "hello world"


def test_extract_user_text_no_parts(_patch_a2a) -> None:
    from ovos_persona_server.a2a import OVOSPersonaAgentExecutor

    msg = MagicMock()
    msg.parts = []
    assert OVOSPersonaAgentExecutor._extract_user_text(msg) == ""


# ---------------------------------------------------------------------------
# _agent_card
# ---------------------------------------------------------------------------

def test_agent_card_uses_persona_name(_patch_a2a) -> None:
    a2a_mod = _patch_a2a
    persona = _fake_persona([])
    persona.name = "my-persona"

    card = a2a_mod._agent_card(persona, "http://host:8337/a2a")
    assert card.name == "my-persona"
    assert card.url == "http://host:8337/a2a"


def test_agent_card_uses_config_description(_patch_a2a) -> None:
    a2a_mod = _patch_a2a
    persona = _fake_persona([])
    persona.config = {"description": "A custom description"}

    card = a2a_mod._agent_card(persona, "http://host/a2a")
    assert card.description == "A custom description"


def test_agent_card_fallback_description(_patch_a2a) -> None:
    a2a_mod = _patch_a2a
    persona = _fake_persona([])
    persona.name = "mypersona"
    persona.config = {}

    card = a2a_mod._agent_card(persona, "http://host/a2a")
    assert "mypersona" in card.description


# ---------------------------------------------------------------------------
# OVOSPersonaAgentExecutor.execute
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_emits_artifact_and_status(_patch_a2a) -> None:
    from ovos_persona_server.a2a import OVOSPersonaAgentExecutor

    persona = _fake_persona(["Hello", "there"])
    executor = OVOSPersonaAgentExecutor(persona)
    queue = _FakeEventQueue()
    ctx = _FakeRequestContext(_FakeMessage("hi"))

    await executor.execute(ctx, queue)

    artifact_events = [e for e in queue.events if isinstance(e, _FakeTaskArtifactUpdateEvent)]
    status_events = [e for e in queue.events if isinstance(e, _FakeTaskStatusUpdateEvent)]

    assert len(artifact_events) == 2
    assert len(status_events) == 1
    assert status_events[0].status.state == _FakeTaskState.completed
    assert status_events[0].final is True


@pytest.mark.asyncio
async def test_execute_skips_empty_chunks(_patch_a2a) -> None:
    from ovos_persona_server.a2a import OVOSPersonaAgentExecutor

    persona = _fake_persona(["Hello", "", "world"])
    executor = OVOSPersonaAgentExecutor(persona)
    queue = _FakeEventQueue()
    ctx = _FakeRequestContext(_FakeMessage("hi"))

    await executor.execute(ctx, queue)

    artifact_events = [e for e in queue.events if isinstance(e, _FakeTaskArtifactUpdateEvent)]
    assert len(artifact_events) == 2  # empty chunk skipped


@pytest.mark.asyncio
async def test_execute_last_chunk_flag(_patch_a2a) -> None:
    from ovos_persona_server.a2a import OVOSPersonaAgentExecutor

    persona = _fake_persona(["A", "B", "C"])
    executor = OVOSPersonaAgentExecutor(persona)
    queue = _FakeEventQueue()
    ctx = _FakeRequestContext(_FakeMessage("hi"))

    await executor.execute(ctx, queue)

    artifact_events = [e for e in queue.events if isinstance(e, _FakeTaskArtifactUpdateEvent)]
    assert artifact_events[-1].last_chunk is True
    for ev in artifact_events[:-1]:
        assert ev.last_chunk is False


@pytest.mark.asyncio
async def test_execute_forwards_user_text_to_persona(_patch_a2a) -> None:
    from ovos_persona_server.a2a import OVOSPersonaAgentExecutor

    persona = _fake_persona(["ok"])
    executor = OVOSPersonaAgentExecutor(persona)
    queue = _FakeEventQueue()
    ctx = _FakeRequestContext(_FakeMessage("tell me a joke"))

    await executor.execute(ctx, queue)

    # run_stream passes a Session positionally alongside the messages.
    persona.stream.assert_called_once()
    assert persona.stream.call_args.args[0] == [{"role": "user", "content": "tell me a joke"}]


# ---------------------------------------------------------------------------
# OVOSPersonaAgentExecutor.cancel
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_emits_canceled_status(_patch_a2a) -> None:
    from ovos_persona_server.a2a import OVOSPersonaAgentExecutor

    persona = _fake_persona([])
    executor = OVOSPersonaAgentExecutor(persona)
    queue = _FakeEventQueue()
    ctx = _FakeRequestContext(_FakeMessage(""))

    await executor.cancel(ctx, queue)

    assert len(queue.events) == 1
    ev = queue.events[0]
    assert isinstance(ev, _FakeTaskStatusUpdateEvent)
    assert ev.status.state == _FakeTaskState.canceled
    assert ev.final is True


# ---------------------------------------------------------------------------
# create_a2a_application
# ---------------------------------------------------------------------------

def test_create_a2a_application_raises_without_sdk(monkeypatch) -> None:
    import ovos_persona_server.a2a as a2a_mod
    monkeypatch.setattr(a2a_mod, "_A2A_AVAILABLE", False)

    persona = _fake_persona([])
    with pytest.raises(RuntimeError, match="a2a-sdk is not installed"):
        a2a_mod.create_a2a_application(persona)


def test_create_a2a_application_returns_starlette_app(_patch_a2a, monkeypatch) -> None:
    a2a_mod = _patch_a2a

    class _FakeA2AApp:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def build(self) -> MagicMock:
            return MagicMock()

    monkeypatch.setattr(a2a_mod, "A2AStarletteApplication", _FakeA2AApp)
    monkeypatch.setattr(a2a_mod, "DefaultRequestHandler", MagicMock())
    monkeypatch.setattr(a2a_mod, "InMemoryTaskStore", MagicMock())

    persona = _fake_persona([])
    app = a2a_mod.create_a2a_application(persona, "http://host/a2a")
    assert isinstance(app, _FakeA2AApp)
