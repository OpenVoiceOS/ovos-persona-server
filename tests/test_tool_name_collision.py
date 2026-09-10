"""A client tool must not be shadowed by a persona tool of the same name.

Client-side and server-side tools are merged into one `offered` list and
dispatch is decided by membership in the server's name set. Nothing dedupes
that list, so when a client sends a tool whose name matches one of the
persona's ToolBox tools -- easy with generic names like "search" -- two things
go wrong at once: the model is shown two entries with the same name, and the
call is executed server-side instead of being relayed for the client to run.
"""
import types

import pytest

from ovos_persona_server import server_tools
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole, ToolCall


class _Tool:
    description = "server side search"

    class argument_schema:
        @staticmethod
        def model_json_schema():
            return {"type": "object", "properties": {}}


def _registry():
    return {"search": (object(), _Tool())}


class _Engine:
    """Requests `search` once, then answers. Records what it was offered."""

    supports_tools = True

    def __init__(self):
        self.offered = []
        self.calls = 0

    def continue_chat(self, messages, session_id=None, lang=None, units=None, tools=None):
        self.offered.append(tools or [])
        self.calls += 1
        if self.calls == 1:
            return AgentMessage(role=MessageRole.ASSISTANT, content="",
                                tool_calls=[ToolCall(id="1", name="search",
                                                     arguments={})])
        return AgentMessage(role=MessageRole.ASSISTANT, content="done")


CLIENT_SPEC = {"type": "function",
               "function": {"name": "search", "description": "client side search",
                            "parameters": {"type": "object", "properties": {}}}}


def test_duplicate_tool_name_is_not_offered_twice(monkeypatch):
    monkeypatch.setattr(server_tools, "invoke_tool",
                        lambda name, args, registry=None: "server ran it")
    engine = _Engine()
    server_tools.run_tool_loop(engine, [{"role": "user", "content": "hi"}],
                               client_specs=[CLIENT_SPEC], registry=_registry())
    names = [t["function"]["name"] for t in engine.offered[0]]
    assert names.count("search") == 1, (
        f"the model was offered the same tool name twice: {names}")
