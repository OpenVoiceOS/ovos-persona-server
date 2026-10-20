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
Tests for the A2A server adapter (ovos_persona_server/a2a.py) against the
REAL a2a-sdk (no mocked SDK internals). ``a2a-sdk`` is an optional
dependency (the ``[a2a]`` extra); if it is not installed the whole module
is skipped rather than silently passing.

Covers:
  - _A2A_AVAILABLE=False path (RuntimeError + warning log), forced by
    simulating an absent SDK via sys.modules manipulation.
  - _agent_card content built from a loaded persona.
  - OVOSPersonaAgentExecutor.execute/cancel happy paths against the real
    a2a-sdk event/task machinery.
  - A genuine round trip through the mounted ASGI app: a real A2A client
    sends a message over an in-process ASGI transport and gets the
    persona's streamed response back.
"""

import asyncio
import importlib
import logging
import sys
from unittest.mock import MagicMock

import pytest

pytest.importorskip("a2a", reason="a2a-sdk (the [a2a] extra) is not installed")

import httpx
from fastapi import FastAPI

import ovos_persona_server.a2a as a2a_mod

# a2a-sdk is installed, so it must load. a2a.py starts the server without A2A
# when the SDK fails inside its own import; a skip here would hide that
# failure behind a green run.
assert a2a_mod._A2A_AVAILABLE, (
    "a2a-sdk is installed but ovos_persona_server.a2a could not import it")

from a2a.server.events import EventQueueLegacy  # noqa: E402


def _fake_persona(sentences, name="test-persona", description=None):
    persona = MagicMock()
    persona.name = name
    persona.config = {"description": description} if description else {}
    persona.stream.return_value = iter(sentences)
    return persona


# ---------------------------------------------------------------------------
# _A2A_AVAILABLE = False path
# ---------------------------------------------------------------------------

def _block_a2a_sdk(monkeypatch) -> None:
    """Make every ``a2a`` import fail until the test ends.

    The SDK modules are shadowed with ``None`` rather than deleted. Deleting
    them makes the next import load a second copy of the SDK, and the SDK's
    own isinstance checks then fail in every later test of the process.
    """
    for key in [k for k in sys.modules if k == "a2a" or k.startswith("a2a.")]:
        monkeypatch.setitem(sys.modules, key, None)
    monkeypatch.setitem(sys.modules, "a2a", None)
    monkeypatch.delitem(sys.modules, "ovos_persona_server.a2a", raising=False)
    # The re-import rebinds the package attribute, so restore that too.
    monkeypatch.setattr(sys.modules["ovos_persona_server"], "a2a", a2a_mod)


class TestA2ANotAvailable:
    def test_create_a2a_application_raises_runtime_error_when_sdk_missing(self, monkeypatch):
        """create_a2a_application must raise RuntimeError when a2a-sdk absent."""
        _block_a2a_sdk(monkeypatch)
        mod = importlib.import_module("ovos_persona_server.a2a")
        assert mod._A2A_AVAILABLE is False
        with pytest.raises(RuntimeError, match="a2a-sdk"):
            mod.create_a2a_application(_fake_persona([]))

    def test_missing_sdk_warning_logged(self, monkeypatch, caplog):
        _block_a2a_sdk(monkeypatch)
        with caplog.at_level(logging.WARNING):
            mod = importlib.import_module("ovos_persona_server.a2a")
        assert mod._A2A_AVAILABLE is False
        assert any("a2a" in r.message.lower() for r in caplog.records)


_FAIL_INSIDE_A2A = """
import importlib.abc, importlib.machinery, sys

class _Broken(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    # Stands in for an a2a-sdk that raises inside its own import, as
    # a2a-sdk 1.2.3 does on Python 3.12 with httpx 1.0.dev6.
    def find_spec(self, name, path, target=None):
        if name == "a2a.server.tasks":
            return importlib.machinery.ModuleSpec(name, self)
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        raise AttributeError("module 'httpx' has no attribute 'AsyncClient'")

sys.meta_path.insert(0, _Broken())
"""


class TestA2ASdkBrokenAtImport:
    """An installed a2a-sdk that fails inside its import must not stop the
    server: A2A is optional, so the server logs why and starts without it."""

    def _run(self, tmp_path, code):
        import subprocess
        persona = tmp_path / "p.json"
        persona.write_text('{"name": "p", "solvers": []}')
        script = _FAIL_INSIDE_A2A + code.format(persona=str(persona))
        return subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, timeout=120)

    def test_module_import_logs_and_disables_a2a(self, tmp_path):
        proc = self._run(tmp_path, "import ovos_persona_server.a2a as a\n"
                                   "assert a._A2A_AVAILABLE is False\n")
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "AttributeError: module 'httpx' has no attribute 'AsyncClient'" in proc.stderr

    def test_server_starts_without_a2a(self, tmp_path):
        proc = self._run(tmp_path, (
            "import ovos_persona_server\n"
            "from types import SimpleNamespace\n"
            "ovos_persona_server.Persona = lambda n, c: SimpleNamespace(name=n, config=c)\n"
            "app = ovos_persona_server.create_persona_app("
            "[{persona!r}], a2a_base_url='http://127.0.0.1:1/a2a')\n"
            "assert not any(getattr(r, 'path', '').startswith('/a2a') for r in app.routes)\n"))
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "A2A endpoint will not be available" in proc.stderr


# ---------------------------------------------------------------------------
# Agent card content
# ---------------------------------------------------------------------------

class TestAgentCardSpecCompliance:
    """A2A 1.0 marks AgentInterface.protocolVersion and AgentSkill.tags as
    REQUIRED, and section 5.7 says a required array MUST contain at least one
    element."""

    def _card(self):
        return a2a_mod._agent_card(_fake_persona([]), "http://localhost:8337/a2a")

    def test_card_passes_the_sdk_strict_required_field_check(self):
        from a2a.utils.proto_utils import validate_proto_required_fields
        validate_proto_required_fields(self._card())

    def test_routes_log_no_spec_compliance_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            a2a_mod.create_a2a_application(_fake_persona([]), "http://localhost:8337/a2a")
        assert not [r.message for r in caplog.records if "not spec-compliant" in r.message]

    def test_interfaces_declare_1_0_then_0_3_on_one_endpoint(self):
        interfaces = [(i.url, i.protocol_binding, i.protocol_version)
                      for i in self._card().supported_interfaces]
        assert interfaces == [("http://localhost:8337/a2a/", "JSONRPC", "1.0"),
                              ("http://localhost:8337/a2a/", "JSONRPC", "0.3")]

    def test_served_card_keeps_the_0_3_fields_for_older_clients(self):
        from a2a.server.request_handlers.response_helpers import agent_card_to_dict
        card = agent_card_to_dict(self._card())
        assert (card["url"], card["preferredTransport"], card["protocolVersion"]) == (
            "http://localhost:8337/a2a/", "JSONRPC", "0.3")


class TestAgentCardContent:
    def test_agent_card_name_matches_persona(self):
        persona = _fake_persona([], name="my-persona")
        card = a2a_mod._agent_card(persona, "http://localhost:8337/a2a")
        assert card.name == "my-persona"

    def test_agent_card_url_has_trailing_slash(self):
        """The JSON-RPC route is registered at rpc_url="/" inside an app
        mounted at the given base path, so the real endpoint has a trailing
        slash. A base_url without one must not be advertised verbatim —
        Starlette 307-redirects "/a2a" -> "/a2a/", and the a2a-sdk transport
        does not follow redirects."""
        persona = _fake_persona([], name="test")
        card = a2a_mod._agent_card(persona, "http://example.com/a2a")
        assert card.supported_interfaces[0].url == "http://example.com/a2a/"

    def test_agent_card_url_does_not_double_slash(self):
        persona = _fake_persona([], name="test")
        card = a2a_mod._agent_card(persona, "http://example.com/a2a/")
        assert card.supported_interfaces[0].url == "http://example.com/a2a/"

    def test_agent_card_capabilities_streaming_enabled(self):
        persona = _fake_persona([], name="test")
        card = a2a_mod._agent_card(persona, "http://x/")
        assert card.capabilities.streaming is True

    def test_agent_card_default_modes_are_text(self):
        persona = _fake_persona([], name="test")
        card = a2a_mod._agent_card(persona, "http://x/")
        assert list(card.default_input_modes) == ["text"]
        assert list(card.default_output_modes) == ["text"]

    def test_agent_card_description_from_config(self):
        persona = _fake_persona([], name="test", description="A very helpful bot.")
        card = a2a_mod._agent_card(persona, "http://x")
        assert "helpful bot" in card.description

    def test_agent_card_description_fallback(self):
        persona = _fake_persona([], name="my-bot")
        card = a2a_mod._agent_card(persona, "http://x")
        assert "my-bot" in card.description

    def test_agent_card_has_skills(self):
        persona = _fake_persona([], name="test")
        card = a2a_mod._agent_card(persona, "http://x")
        assert len(card.skills) >= 1
        assert card.skills[0].id == "chat"

    def test_agent_card_skill_has_tags(self):
        """A2A 1.0 marks AgentSkill.tags REQUIRED, and a required array MUST
        contain at least one element."""
        persona = _fake_persona([], name="test")
        card = a2a_mod._agent_card(persona, "http://x")
        assert list(card.skills[0].tags) == ["chat", "conversation"]


# ---------------------------------------------------------------------------
# OVOSPersonaAgentExecutor — real EventQueue, real TaskUpdater
# ---------------------------------------------------------------------------

class TestOVOSPersonaAgentExecutor:
    def _make_context(self, text="Hello", task_id="task-1", context_id="ctx-1"):
        ctx = MagicMock()
        ctx.task_id = task_id
        ctx.context_id = context_id
        ctx.get_user_input.return_value = text
        return ctx

    def test_execute_calls_persona_stream_with_user_text(self):
        from ovos_plugin_manager.templates.agents import MessageRole

        persona = _fake_persona(["Hello ", "World"])
        executor = a2a_mod.OVOSPersonaAgentExecutor(persona)
        queue = EventQueueLegacy()
        ctx = self._make_context("Tell me a story.")

        asyncio.run(executor.execute(ctx, queue))

        persona.stream.assert_called_once()
        sent = persona.stream.call_args[0][0]
        assert sent[0].role == MessageRole.USER
        assert "Tell me a story." in sent[0].content

    def test_execute_emits_completed_task(self):
        persona = _fake_persona(["hi"])
        executor = a2a_mod.OVOSPersonaAgentExecutor(persona)
        queue = EventQueueLegacy()
        ctx = self._make_context()

        asyncio.run(executor.execute(ctx, queue))

        events = []
        while not queue.queue.empty():
            events.append(queue.queue.get_nowait())

        # Task, then a WORKING status update, then an artifact update, then
        # a COMPLETED status update.
        assert any(
            getattr(e, "status", None) is not None
            and e.status.state == a2a_mod.TaskState.TASK_STATE_COMPLETED
            for e in events
        )

    def test_execute_empty_chunks_skipped(self):
        persona = _fake_persona(["", "hello", ""])
        executor = a2a_mod.OVOSPersonaAgentExecutor(persona)
        queue = EventQueueLegacy()
        ctx = self._make_context()

        asyncio.run(executor.execute(ctx, queue))

        events = []
        while not queue.queue.empty():
            events.append(queue.queue.get_nowait())
        artifact_events = [e for e in events if hasattr(e, "artifact")]
        assert len(artifact_events) == 1
        assert artifact_events[0].artifact.parts[0].text == "hello"

    def test_cancel_emits_canceled_status(self):
        persona = _fake_persona([])
        executor = a2a_mod.OVOSPersonaAgentExecutor(persona)
        queue = EventQueueLegacy()
        ctx = self._make_context()

        asyncio.run(executor.cancel(ctx, queue))

        events = []
        while not queue.queue.empty():
            events.append(queue.queue.get_nowait())
        assert len(events) == 1
        assert events[0].status.state == a2a_mod.TaskState.TASK_STATE_CANCELED


# ---------------------------------------------------------------------------
# create_a2a_application — real round trip over an in-process ASGI transport
# ---------------------------------------------------------------------------

class TestCreateA2AApplication:
    def test_returns_mountable_starlette_app(self):
        persona = _fake_persona([], name="test-persona")
        app = a2a_mod.create_a2a_application(persona, "http://localhost:9999/a2a")
        assert app is not None
        assert hasattr(app, "agent_card")
        assert app.agent_card.name == "test-persona"

    def test_real_round_trip_send_message(self):
        """Send a real A2A JSON-RPC message/send request through the mounted
        app via an in-process ASGI transport and get the persona's streamed
        response back — not just an import/shape check.

        No ``follow_redirects=True`` here: the a2a-sdk's own transport does
        not follow redirects either, so if the agent card advertised a URL
        Starlette would 307 on, this test must fail the same way a real
        remote client would."""
        from a2a.client import ClientConfig, create_client
        from a2a.types import Message, Part, Role, SendMessageRequest

        persona = _fake_persona(["Hello ", "there!"], name="roundtrip-persona")
        a2a_app = a2a_mod.create_a2a_application(persona, "http://testserver/a2a")

        fastapi_app = FastAPI()
        fastapi_app.mount("/a2a", a2a_app)

        async def _run():
            transport = httpx.ASGITransport(app=fastapi_app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as hc:
                client = await create_client(
                    a2a_app.agent_card,
                    client_config=ClientConfig(httpx_client=hc, streaming=False),
                )
                msg = Message(
                    role=Role.ROLE_USER,
                    parts=[Part(text="hi there")],
                    message_id="m1",
                )
                req = SendMessageRequest(message=msg)
                results = []
                async for event in client.send_message(req):
                    results.append(event)
                return results

        events = asyncio.run(_run())

        assert len(events) == 1
        task = events[0].task
        assert task.status.state == a2a_mod.TaskState.TASK_STATE_COMPLETED
        assert len(task.artifacts) == 1
        text = "".join(p.text for p in task.artifacts[0].parts)
        assert text == "Hello there!"

        persona.stream.assert_called_once()
        sent = persona.stream.call_args[0][0]
        assert sent[0].content == "hi there"

    def test_agent_card_available_at_both_well_known_paths(self):
        """The 1.x well-known path and the pre-1.0 (0.3.x) one both serve
        the card, so already-deployed 0.3.x clients can still discover this
        agent even though a fresh install cannot reach the 0.3.x SDK."""
        persona = _fake_persona([], name="discoverable-persona")
        a2a_app = a2a_mod.create_a2a_application(persona, "http://testserver/a2a")

        fastapi_app = FastAPI()
        fastapi_app.mount("/a2a", a2a_app)

        async def _run():
            transport = httpx.ASGITransport(app=fastapi_app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as hc:
                new_path = await hc.get("/a2a/.well-known/agent-card.json")
                old_path = await hc.get("/a2a/.well-known/agent.json")
                return new_path, old_path

        new_resp, old_resp = asyncio.run(_run())
        assert new_resp.status_code == 200
        assert old_resp.status_code == 200
        assert new_resp.json()["name"] == "discoverable-persona"
        assert old_resp.json()["name"] == "discoverable-persona"

    def test_v0_3_message_send_method_name_accepted(self):
        """enable_v0_3_compat=True means a 0.3.x client sending the old
        ``message/send`` JSON-RPC method name is served, not rejected with
        -32009/-32601 the way it would be against a bare 1.x route."""
        persona = _fake_persona(["hi"], name="compat-persona")
        a2a_app = a2a_mod.create_a2a_application(persona, "http://testserver/a2a")

        fastapi_app = FastAPI()
        fastapi_app.mount("/a2a", a2a_app)

        async def _run():
            transport = httpx.ASGITransport(app=fastapi_app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as hc:
                return await hc.post(
                    "/a2a/",
                    json={
                        "jsonrpc": "2.0",
                        "id": "1",
                        "method": "message/send",
                        "params": {
                            "message": {
                                "role": "user",
                                "parts": [{"kind": "text", "text": "hi there"}],
                                "messageId": "m1",
                            }
                        },
                    },
                )

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        body = resp.json()
        assert "error" not in body, body


class TestArtifactSentenceSpacing:
    """Sentence chunks are appended by the client, so the server must keep the
    space between them without doubling one that is already there."""

    @staticmethod
    def _artifact_text(sentences):
        from a2a.client import ClientConfig, create_client
        from a2a.types import Message, Part, Role, SendMessageRequest

        persona = _fake_persona(sentences, name="spacing-persona")
        a2a_app = a2a_mod.create_a2a_application(persona, "http://testserver/a2a")
        fastapi_app = FastAPI()
        fastapi_app.mount("/a2a", a2a_app)

        async def _run():
            transport = httpx.ASGITransport(app=fastapi_app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as hc:
                client = await create_client(
                    a2a_app.agent_card,
                    client_config=ClientConfig(httpx_client=hc, streaming=False),
                )
                req = SendMessageRequest(message=Message(
                    role=Role.ROLE_USER, parts=[Part(text="who")], message_id="m1"))
                return [e async for e in client.send_message(req)]

        task = asyncio.run(_run())[0].task
        return "".join(p.text for p in task.artifacts[0].parts)

    def test_two_sentences_are_separated_by_one_space(self):
        text = self._artifact_text(["She is Diana.", "This is her story."])
        assert text == "She is Diana. This is her story."

    def test_trailing_space_is_not_doubled(self):
        text = self._artifact_text(["She is Diana. ", "This is her story."])
        assert text == "She is Diana. This is her story."

    def test_leading_space_is_not_doubled(self):
        text = self._artifact_text(["She is Diana.", " This is her story."])
        assert text == "She is Diana. This is her story."

    def test_empty_chunks_add_no_space(self):
        text = self._artifact_text(["One.", "", "Two."])
        assert text == "One. Two."
