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
"""
Focused unit tests for ``OVOSPersonaAgentExecutor`` internals, isolating the
persona-facing behaviour (stream forwarding, chunk batching, artifact
append/last_chunk flags) from the A2A wire protocol. Unlike
``tests/test_a2a_server.py`` this file mocks the ``EventQueue`` (to assert on
individual enqueued events without draining an asyncio.Queue) but always
imports the REAL a2a-sdk types (``a2a.types.Part``/``TaskState``/etc.) — no
SDK internals are faked, so a shape change in the SDK breaks these tests
immediately instead of being silently absorbed by a hand-rolled stub.
"""

from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("a2a", reason="a2a-sdk (the [a2a] extra) is not installed")

import ovos_persona_server.a2a as a2a_mod

pytestmark = pytest.mark.skipif(
    not a2a_mod._A2A_AVAILABLE,
    reason="a2a-sdk imported but ovos_persona_server.a2a could not wire it up",
)


def _fake_persona(sentences: List[str]) -> MagicMock:
    """Return a mock Persona whose stream() returns *sentences*."""
    persona = MagicMock()
    persona.name = "test-persona"
    persona.config = {}
    persona.stream.return_value = iter(sentences)
    return persona


def _fake_context(text: str = "hi", task_id: str = "t1", context_id: str = "c1") -> Any:
    ctx = MagicMock()
    ctx.task_id = task_id
    ctx.context_id = context_id
    ctx.get_user_input.return_value = text
    return ctx


def _fake_queue() -> MagicMock:
    queue = MagicMock()
    queue.enqueue_event = AsyncMock()
    return queue


# ---------------------------------------------------------------------------
# _agent_card
# ---------------------------------------------------------------------------

def test_agent_card_uses_persona_name() -> None:
    persona = _fake_persona([])
    persona.name = "my-persona"

    card = a2a_mod._agent_card(persona, "http://host:8337/a2a")
    assert card.name == "my-persona"
    # Trailing slash: the JSON-RPC route lives at rpc_url="/" inside the app
    # mounted at this base path, so the real endpoint is ".../a2a/", not
    # ".../a2a" (Starlette 307s the latter and the a2a-sdk transport does
    # not follow redirects).
    assert card.supported_interfaces[0].url == "http://host:8337/a2a/"


def test_agent_card_uses_config_description() -> None:
    persona = _fake_persona([])
    persona.config = {"description": "A custom description"}

    card = a2a_mod._agent_card(persona, "http://host/a2a")
    assert card.description == "A custom description"


def test_agent_card_fallback_description() -> None:
    persona = _fake_persona([])
    persona.name = "mypersona"
    persona.config = {}

    card = a2a_mod._agent_card(persona, "http://host/a2a")
    assert "mypersona" in card.description


# ---------------------------------------------------------------------------
# OVOSPersonaAgentExecutor.execute
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_emits_task_then_status_updates_then_completes() -> None:
    persona = _fake_persona(["Hello", "there"])
    executor = a2a_mod.OVOSPersonaAgentExecutor(persona)
    queue = _fake_queue()
    ctx = _fake_context("hi")

    await executor.execute(ctx, queue)

    events = [c.args[0] for c in queue.enqueue_event.call_args_list]

    # Task, WORKING status, one artifact-update event per chunk (both chunks
    # belong to the same artifact_id="response", append=False then True),
    # then a COMPLETED status.
    task_events = [e for e in events if isinstance(e, a2a_mod.Task)]
    artifact_events = [e for e in events if hasattr(e, "artifact")]
    status_events = [
        e for e in events if hasattr(e, "status") and not isinstance(e, a2a_mod.Task)
    ]

    assert len(task_events) == 1
    assert len(artifact_events) == 2
    assert {a.artifact.artifact_id for a in artifact_events} == {"response"}
    assert status_events[-1].status.state == a2a_mod.TaskState.TASK_STATE_COMPLETED


@pytest.mark.asyncio
async def test_execute_skips_empty_chunks() -> None:
    persona = _fake_persona(["Hello", "", "world"])
    executor = a2a_mod.OVOSPersonaAgentExecutor(persona)
    queue = _fake_queue()
    ctx = _fake_context("hi")

    await executor.execute(ctx, queue)

    events = [c.args[0] for c in queue.enqueue_event.call_args_list]
    artifact_events = [e for e in events if hasattr(e, "artifact")]
    # Both non-empty chunks are batched into the same artifact_id="response"
    # artifact (append=False then append=True) — still 2 update events.
    assert len(artifact_events) == 2
    assert artifact_events[0].append is False
    assert artifact_events[1].append is True


@pytest.mark.asyncio
async def test_execute_no_answer_emits_failed_status() -> None:
    """A persona that streams nothing usable must terminate the task, not crash execute()."""
    persona = _fake_persona(["", ""])
    executor = a2a_mod.OVOSPersonaAgentExecutor(persona)
    queue = _fake_queue()
    ctx = _fake_context("hi")

    await executor.execute(ctx, queue)  # must not raise

    events = [c.args[0] for c in queue.enqueue_event.call_args_list]
    artifact_events = [e for e in events if hasattr(e, "artifact")]
    status_events = [
        e for e in events if hasattr(e, "status") and not isinstance(e, a2a_mod.Task)
    ]

    assert len(artifact_events) == 0
    assert status_events
    assert status_events[-1].status.state == a2a_mod.TaskState.TASK_STATE_FAILED


@pytest.mark.asyncio
async def test_execute_last_chunk_flag() -> None:
    persona = _fake_persona(["A", "B", "C"])
    executor = a2a_mod.OVOSPersonaAgentExecutor(persona)
    queue = _fake_queue()
    ctx = _fake_context("hi")

    await executor.execute(ctx, queue)

    events = [c.args[0] for c in queue.enqueue_event.call_args_list]
    artifact_events = [e for e in events if hasattr(e, "artifact")]
    assert artifact_events[-1].last_chunk is True
    for ev in artifact_events[:-1]:
        assert ev.last_chunk is False


@pytest.mark.asyncio
async def test_execute_forwards_user_text_to_persona() -> None:
    from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

    persona = _fake_persona(["ok"])
    executor = a2a_mod.OVOSPersonaAgentExecutor(persona)
    queue = _fake_queue()
    ctx = _fake_context("tell me a joke")

    await executor.execute(ctx, queue)

    # run_stream passes a Session positionally alongside the messages.
    persona.stream.assert_called_once()
    sent = persona.stream.call_args.args[0]
    assert len(sent) == 1
    assert isinstance(sent[0], AgentMessage)
    assert sent[0].role == MessageRole.USER
    assert sent[0].content == "tell me a joke"


# ---------------------------------------------------------------------------
# OVOSPersonaAgentExecutor.cancel
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_emits_canceled_status() -> None:
    persona = _fake_persona([])
    executor = a2a_mod.OVOSPersonaAgentExecutor(persona)
    queue = _fake_queue()
    ctx = _fake_context("")

    await executor.cancel(ctx, queue)

    queue.enqueue_event.assert_called_once()
    ev = queue.enqueue_event.call_args[0][0]
    assert ev.status.state == a2a_mod.TaskState.TASK_STATE_CANCELED


# ---------------------------------------------------------------------------
# create_a2a_application
# ---------------------------------------------------------------------------

def test_create_a2a_application_raises_without_sdk(monkeypatch) -> None:
    monkeypatch.setattr(a2a_mod, "_A2A_AVAILABLE", False)

    persona = _fake_persona([])
    with pytest.raises(RuntimeError, match="a2a-sdk is not installed"):
        a2a_mod.create_a2a_application(persona)


def test_create_a2a_application_returns_starlette_app() -> None:
    persona = _fake_persona([])
    persona.name = "test-persona"

    app = a2a_mod.create_a2a_application(persona, "http://host/a2a")
    assert isinstance(app, a2a_mod.Starlette)
    assert app.agent_card.name == "test-persona"


# ---------------------------------------------------------------------------
# memory session key (CHAT_MEMORY=transparent)
#
# ``context_id`` is the A2A conversation identifier — per the protocol it is the
# id of the "contextual collection of interactions (tasks and messages)", so it
# is stable across the turns of one conversation and distinct between callers.
# Ported from the pre-1.x mocked test file onto the real ``EventQueueLegacy``
# and a ``RequestContext``-shaped mock (``get_user_input``/``context_id``).
# ---------------------------------------------------------------------------

class _RecordingMemory:
    """Memory plugin double that records which session key it was asked for."""

    def __init__(self) -> None:
        self.history: dict = {}
        self.keys: List[str] = []

    def build_conversation_context(self, utterance, session_id):
        self.keys.append(session_id)
        return list(self.history.get(session_id, [])) + [utterance]

    def update_history(self, new_messages, session_id):
        self.keys.append(session_id)
        self.history.setdefault(session_id, []).extend(new_messages)


@pytest.fixture()
def _transparent_memory(monkeypatch):
    from ovos_persona_server import persona as persona_mod
    monkeypatch.setattr(persona_mod.settings, "chat_memory", "transparent")


@pytest.mark.asyncio
async def test_execute_keys_memory_on_context_id(_transparent_memory) -> None:
    """Two conversations must not share a memory bucket."""
    memory = _RecordingMemory()
    persona = _fake_persona(["ok"])
    persona.memory = memory
    executor = a2a_mod.OVOSPersonaAgentExecutor(persona)

    await executor.execute(
        _fake_context("the code is hunter2", context_id="conv-a"), _fake_queue()
    )
    persona.stream.return_value = iter(["ok"])
    await executor.execute(
        _fake_context("what is the code", context_id="conv-b"), _fake_queue()
    )

    assert set(memory.history) == {"conv-a", "conv-b"}
    assert "hunter2" not in str(memory.history["conv-b"])


@pytest.mark.asyncio
async def test_execute_reuses_bucket_within_one_conversation(_transparent_memory) -> None:
    """Same context_id across turns keeps one bucket, so memory still works."""
    memory = _RecordingMemory()
    persona = _fake_persona(["ok"])
    persona.memory = memory
    executor = a2a_mod.OVOSPersonaAgentExecutor(persona)

    await executor.execute(
        _fake_context("the code is hunter2", context_id="conv-a"), _fake_queue()
    )
    persona.stream.return_value = iter(["ok"])
    await executor.execute(
        _fake_context("what is the code", context_id="conv-a"), _fake_queue()
    )

    assert list(memory.history) == ["conv-a"]
    # the second turn was built on top of the first, not on an empty bucket
    assert "hunter2" in str(memory.history["conv-a"])


@pytest.mark.asyncio
async def test_a_client_that_sends_no_context_id_gets_a_fresh_bucket(
        _transparent_memory) -> None:
    """A caller that does not echo contextId starts a new conversation each time.

    The a2a-sdk mints a fresh identifier for a request that carries none, so
    there is nothing stable to key history on. That is the anonymous rule --
    an unidentified caller is independent, not merged into a shared bucket --
    but it means continuity on this surface requires the client to echo the id
    back. Pinned deliberately so a later change does not "fix" this into one
    shared bucket, which is the leak the transparent mode is documented to
    avoid.
    """
    memory = _RecordingMemory()
    persona = _fake_persona(["ok"])
    persona.memory = memory
    executor = a2a_mod.OVOSPersonaAgentExecutor(persona)

    await executor.execute(
        _fake_context("the code is hunter2", context_id="generated-1"), _fake_queue()
    )
    persona.stream.return_value = iter(["ok"])
    await executor.execute(
        _fake_context("what is the code", context_id="generated-2"), _fake_queue()
    )

    assert sorted(memory.history) == ["generated-1", "generated-2"]
    assert "hunter2" not in str(memory.history["generated-2"])
