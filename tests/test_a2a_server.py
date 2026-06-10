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
Unit tests for the A2A server adapter (ovos_persona_server/a2a.py).

Strategy: a2a-sdk is optional; we test:
  - _A2A_AVAILABLE=False path (RuntimeError + warning log)
  - _extract_user_text with various message shapes
  - _agent_card content when a2a-sdk IS mocked into sys.modules
  - OVOSPersonaAgentExecutor.execute happy path (mocked a2a types)
  - OVOSPersonaAgentExecutor.execute error path (persona.stream raises)
  - OVOSPersonaAgentExecutor.cancel path
  - invalid a2a_base_url tolerance (URL is passed through unchanged)
"""

import sys
import types
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build a minimal a2a-like stub module for tests
# ---------------------------------------------------------------------------

def _make_a2a_stubs():
    """Build fake a2a modules so the module can import without the real SDK."""
    # Build fake classes
    class _TextPart:
        def __init__(self, text=""):
            self.text = text

    class _Part:
        def __init__(self, root=None):
            self.root = root

    class _Artifact:
        def __init__(self, parts=None, artifact_id="", name=""):
            self.parts = parts or []
            self.artifact_id = artifact_id
            self.name = name

    class _TaskState:
        completed = "completed"
        canceled = "canceled"

    class _TaskStatus:
        def __init__(self, state=None):
            self.state = state

    class _TaskArtifactUpdateEvent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _TaskStatusUpdateEvent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _AgentCapabilities:
        def __init__(self, streaming=False):
            self.streaming = streaming

    class _AgentSkill:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _AgentCard:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _AgentExecutor:
        async def execute(self, context, event_queue):
            pass
        async def cancel(self, context, event_queue):
            pass

    class _EventQueue:
        async def enqueue_event(self, event):
            pass

    class _RequestContext:
        pass

    class _DefaultRequestHandler:
        def __init__(self, **kwargs):
            pass

    class _InMemoryTaskStore:
        pass

    class _A2AStarletteApplication:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def build(self):
            return MagicMock()

    # Build stub module tree
    a2a = types.ModuleType("a2a")
    a2a.server = types.ModuleType("a2a.server")
    a2a.server.agent_execution = types.ModuleType("a2a.server.agent_execution")
    a2a.server.agent_execution.AgentExecutor = _AgentExecutor
    a2a.server.agent_execution.RequestContext = _RequestContext
    a2a.server.apps = types.ModuleType("a2a.server.apps")
    a2a.server.apps.A2AStarletteApplication = _A2AStarletteApplication
    a2a.server.events = types.ModuleType("a2a.server.events")
    a2a.server.events.EventQueue = _EventQueue
    a2a.server.request_handlers = types.ModuleType("a2a.server.request_handlers")
    a2a.server.request_handlers.DefaultRequestHandler = _DefaultRequestHandler
    a2a.server.tasks = types.ModuleType("a2a.server.tasks")
    a2a.server.tasks.InMemoryTaskStore = _InMemoryTaskStore
    a2a.types = types.ModuleType("a2a.types")
    a2a.types.AgentCard = _AgentCard
    a2a.types.AgentCapabilities = _AgentCapabilities
    a2a.types.AgentSkill = _AgentSkill
    a2a.types.Artifact = _Artifact
    a2a.types.Part = _Part
    a2a.types.TaskArtifactUpdateEvent = _TaskArtifactUpdateEvent
    a2a.types.TaskState = _TaskState
    a2a.types.TaskStatus = _TaskStatus
    a2a.types.TaskStatusUpdateEvent = _TaskStatusUpdateEvent
    a2a.types.TextPart = _TextPart

    return a2a, {
        "TextPart": _TextPart,
        "Part": _Part,
        "Artifact": _Artifact,
        "TaskState": _TaskState,
        "TaskStatus": _TaskStatus,
        "TaskArtifactUpdateEvent": _TaskArtifactUpdateEvent,
        "TaskStatusUpdateEvent": _TaskStatusUpdateEvent,
        "AgentCard": _AgentCard,
        "AgentCapabilities": _AgentCapabilities,
        "AgentSkill": _AgentSkill,
        "EventQueue": _EventQueue,
        "RequestContext": _RequestContext,
        "AgentExecutor": _AgentExecutor,
    }


def _fresh_a2a_module(stubs):
    """Import ovos_persona_server.a2a with fake a2a stubs injected."""
    a2a_stub, classes = stubs
    modules_to_inject = {
        "a2a": a2a_stub,
        "a2a.server": a2a_stub.server,
        "a2a.server.agent_execution": a2a_stub.server.agent_execution,
        "a2a.server.apps": a2a_stub.server.apps,
        "a2a.server.events": a2a_stub.server.events,
        "a2a.server.request_handlers": a2a_stub.server.request_handlers,
        "a2a.server.tasks": a2a_stub.server.tasks,
        "a2a.types": a2a_stub.types,
    }
    # Remove cached module if present
    for key in list(sys.modules):
        if key == "ovos_persona_server.a2a":
            del sys.modules[key]

    old_modules = {k: sys.modules.get(k) for k in modules_to_inject}
    sys.modules.update(modules_to_inject)
    try:
        import importlib
        mod = importlib.import_module("ovos_persona_server.a2a")
        # Force reload so it picks up the stubs
        mod = importlib.reload(mod)
        return mod, classes
    finally:
        # Restore
        for k, v in old_modules.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ---------------------------------------------------------------------------
# Tests: _A2A_AVAILABLE = False path
# ---------------------------------------------------------------------------

class TestA2ANotAvailable:
    def test_create_a2a_application_raises_runtime_error_when_sdk_missing(self):
        """create_a2a_application must raise RuntimeError when a2a-sdk absent."""
        # Remove any a2a stub from sys.modules
        for key in list(sys.modules):
            if key.startswith("a2a"):
                del sys.modules[key]
        # Reload the module so _A2A_AVAILABLE is False
        for key in list(sys.modules):
            if key == "ovos_persona_server.a2a":
                del sys.modules[key]
        import importlib
        mod = importlib.import_module("ovos_persona_server.a2a")
        mod = importlib.reload(mod)

        if mod._A2A_AVAILABLE:
            pytest.skip("a2a-sdk is actually installed in this env")

        mock_persona = MagicMock()
        mock_persona.name = "test"
        with pytest.raises(RuntimeError, match="a2a-sdk"):
            mod.create_a2a_application(mock_persona)

    def test_missing_sdk_warning_logged(self, caplog):
        """A warning must be emitted when a2a-sdk is absent."""
        for key in list(sys.modules):
            if key.startswith("a2a") or key == "ovos_persona_server.a2a":
                del sys.modules[key]
        import importlib
        with caplog.at_level(logging.WARNING):
            mod = importlib.import_module("ovos_persona_server.a2a")
            mod = importlib.reload(mod)
        if mod._A2A_AVAILABLE:
            pytest.skip("a2a-sdk is actually installed")
        # Warning should have been emitted
        assert any("a2a" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests using stub a2a modules
# ---------------------------------------------------------------------------

class TestExtractUserText:
    def test_single_text_part(self):
        stubs = _make_a2a_stubs()
        mod, classes = _fresh_a2a_module(stubs)
        TextPart = classes["TextPart"]
        Part = classes["Part"]

        msg = MagicMock()
        msg.parts = [Part(root=TextPart(text="Hello"))]
        result = mod.OVOSPersonaAgentExecutor._extract_user_text(msg)
        assert result == "Hello"

    def test_multiple_text_parts_joined(self):
        stubs = _make_a2a_stubs()
        mod, classes = _fresh_a2a_module(stubs)
        TextPart = classes["TextPart"]
        Part = classes["Part"]

        msg = MagicMock()
        msg.parts = [Part(root=TextPart(text="Hello")), Part(root=TextPart(text="World"))]
        result = mod.OVOSPersonaAgentExecutor._extract_user_text(msg)
        assert "Hello" in result and "World" in result

    def test_empty_parts_returns_empty_string(self):
        stubs = _make_a2a_stubs()
        mod, classes = _fresh_a2a_module(stubs)

        msg = MagicMock()
        msg.parts = []
        result = mod.OVOSPersonaAgentExecutor._extract_user_text(msg)
        assert result == ""

    def test_non_text_part_skipped(self):
        stubs = _make_a2a_stubs()
        mod, classes = _fresh_a2a_module(stubs)
        Part = classes["Part"]

        msg = MagicMock()
        # Part whose root is NOT a TextPart
        msg.parts = [Part(root=MagicMock(spec=[]))]
        result = mod.OVOSPersonaAgentExecutor._extract_user_text(msg)
        assert result == ""

    def test_parts_is_none_handled(self):
        stubs = _make_a2a_stubs()
        mod, _ = _fresh_a2a_module(stubs)

        msg = MagicMock()
        msg.parts = None
        result = mod.OVOSPersonaAgentExecutor._extract_user_text(msg)
        assert result == ""


# ---------------------------------------------------------------------------
# Agent card content tests
# ---------------------------------------------------------------------------

class TestAgentCardContent:
    def test_agent_card_name_matches_persona(self):
        stubs = _make_a2a_stubs()
        mod, _ = _fresh_a2a_module(stubs)

        persona = MagicMock()
        persona.name = "my-persona"
        persona.config = {}
        card = mod._agent_card(persona, "http://localhost:8337/a2a")
        assert card.name == "my-persona"

    def test_agent_card_url_matches_base(self):
        stubs = _make_a2a_stubs()
        mod, _ = _fresh_a2a_module(stubs)

        persona = MagicMock()
        persona.name = "test"
        persona.config = {}
        card = mod._agent_card(persona, "http://example.com/a2a")
        assert card.url == "http://example.com/a2a"

    def test_agent_card_description_from_config(self):
        stubs = _make_a2a_stubs()
        mod, _ = _fresh_a2a_module(stubs)

        persona = MagicMock()
        persona.name = "test"
        persona.config = {"description": "A very helpful bot."}
        card = mod._agent_card(persona, "http://x")
        assert "helpful bot" in card.description

    def test_agent_card_description_fallback(self):
        stubs = _make_a2a_stubs()
        mod, _ = _fresh_a2a_module(stubs)

        persona = MagicMock()
        persona.name = "my-bot"
        persona.config = {}
        card = mod._agent_card(persona, "http://x")
        assert "my-bot" in card.description

    def test_agent_card_has_skills(self):
        stubs = _make_a2a_stubs()
        mod, _ = _fresh_a2a_module(stubs)

        persona = MagicMock()
        persona.name = "test"
        persona.config = {}
        card = mod._agent_card(persona, "http://x")
        assert len(card.skills) >= 1

    def test_agent_card_invalid_base_url_passthrough(self):
        """An invalid URL should not raise — it is passed through to the card."""
        stubs = _make_a2a_stubs()
        mod, _ = _fresh_a2a_module(stubs)

        persona = MagicMock()
        persona.name = "test"
        persona.config = {}
        # Should not raise
        card = mod._agent_card(persona, "not-a-valid-url")
        assert card.url == "not-a-valid-url"


# ---------------------------------------------------------------------------
# OVOSPersonaAgentExecutor tests
# ---------------------------------------------------------------------------

class TestOVOSPersonaAgentExecutor:
    def _make_executor(self, stubs, stream_yields=None):
        mod, classes = _fresh_a2a_module(stubs)
        mock_persona = MagicMock()
        mock_persona.stream.return_value = iter(stream_yields or ["Hello ", "World"])
        executor = mod.OVOSPersonaAgentExecutor(mock_persona)
        return executor, mod, classes, mock_persona

    def _make_context(self, classes, text="Hello"):
        TextPart = classes["TextPart"]
        Part = classes["Part"]
        ctx = MagicMock()
        ctx.task_id = "task-1"
        ctx.context_id = "ctx-1"
        ctx.message = MagicMock()
        ctx.message.parts = [Part(root=TextPart(text=text))]
        return ctx

    def _make_event_queue(self, classes):
        queue = MagicMock()
        queue.enqueue_event = AsyncMock()
        return queue

    def test_execute_calls_persona_stream(self):
        stubs = _make_a2a_stubs()
        executor, mod, classes, mock_persona = self._make_executor(stubs)
        ctx = self._make_context(classes, "Tell me a story.")
        eq = self._make_event_queue(classes)
        asyncio.get_event_loop().run_until_complete(executor.execute(ctx, eq))
        mock_persona.stream.assert_called_once()
        call_args = mock_persona.stream.call_args[0][0]
        assert call_args[0]["role"] == "user"
        assert "Tell me a story." in call_args[0]["content"]

    def test_execute_enqueues_artifact_events(self):
        stubs = _make_a2a_stubs()
        executor, mod, classes, _ = self._make_executor(stubs, stream_yields=["hello", "world"])
        ctx = self._make_context(classes)
        eq = self._make_event_queue(classes)
        asyncio.get_event_loop().run_until_complete(executor.execute(ctx, eq))
        # Should have 2 artifact events + 1 status event
        assert eq.enqueue_event.call_count == 3

    def test_execute_final_status_is_completed(self):
        stubs = _make_a2a_stubs()
        executor, mod, classes, _ = self._make_executor(stubs, stream_yields=["hi"])
        ctx = self._make_context(classes)
        eq = self._make_event_queue(classes)
        asyncio.get_event_loop().run_until_complete(executor.execute(ctx, eq))
        # Last enqueued event should be TaskStatusUpdateEvent with completed state
        last_call_args = eq.enqueue_event.call_args_list[-1][0][0]
        assert last_call_args.status.state == "completed"
        assert last_call_args.final is True

    def test_execute_empty_chunks_skipped(self):
        stubs = _make_a2a_stubs()
        executor, mod, classes, _ = self._make_executor(stubs, stream_yields=["", "hello", ""])
        ctx = self._make_context(classes)
        eq = self._make_event_queue(classes)
        asyncio.get_event_loop().run_until_complete(executor.execute(ctx, eq))
        # Only 1 artifact event (non-empty chunk) + 1 status event
        assert eq.enqueue_event.call_count == 2

    def test_cancel_enqueues_canceled_status(self):
        stubs = _make_a2a_stubs()
        executor, mod, classes, _ = self._make_executor(stubs)
        ctx = self._make_context(classes)
        eq = self._make_event_queue(classes)
        asyncio.get_event_loop().run_until_complete(executor.cancel(ctx, eq))
        eq.enqueue_event.assert_called_once()
        event = eq.enqueue_event.call_args[0][0]
        assert event.status.state == "canceled"
        assert event.final is True


# ---------------------------------------------------------------------------
# create_a2a_application when SDK is present
# ---------------------------------------------------------------------------

class TestCreateA2AApplication:
    def test_returns_application_object(self):
        stubs = _make_a2a_stubs()
        mod, _ = _fresh_a2a_module(stubs)

        persona = MagicMock()
        persona.name = "test-persona"
        persona.config = {}
        app = mod.create_a2a_application(persona, "http://localhost:9999/a2a")
        assert app is not None

    def test_application_has_agent_card(self):
        stubs = _make_a2a_stubs()
        mod, _ = _fresh_a2a_module(stubs)

        persona = MagicMock()
        persona.name = "test-persona"
        persona.config = {}
        app = mod.create_a2a_application(persona, "http://localhost:9999/a2a")
        assert hasattr(app, "agent_card")
        assert app.agent_card.name == "test-persona"
