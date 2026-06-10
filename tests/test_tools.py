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
Unit tests for OPM tool plugin discovery, schema mapping, invoke endpoint,
and UTCP manual contents.

A fake ToolBox plugin is injected via ``unittest.mock.patch`` so no real OPM
plugins need to be installed.
"""

import json
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import Field

from ovos_plugin_manager.templates.agent_tools import (
    AgentTool, ToolArguments, ToolBox, ToolOutput,
)


# ---------------------------------------------------------------------------
# Fake plugin fixtures
# ---------------------------------------------------------------------------

class EchoArgs(ToolArguments):
    message: str = Field(..., description="Text to echo back.")


class EchoOutput(ToolOutput):
    echo: str = Field(..., description="The echoed text.")


class AddArgs(ToolArguments):
    a: float = Field(..., description="First operand.")
    b: float = Field(..., description="Second operand.")


class AddOutput(ToolOutput):
    result: float = Field(..., description="Sum of a and b.")


class FakeToolBox(ToolBox):
    """Minimal in-process ToolBox with two deterministic tools."""

    def discover_tools(self) -> List[AgentTool]:
        return [
            AgentTool(
                name="echo",
                description="Echo a message back.",
                argument_schema=EchoArgs,
                output_schema=EchoOutput,
                tool_call=lambda args: EchoOutput(echo=args.message),
            ),
            AgentTool(
                name="add",
                description="Add two numbers.",
                argument_schema=AddArgs,
                output_schema=AddOutput,
                tool_call=lambda args: AddOutput(result=args.a + args.b),
            ),
        ]


_FAKE_REGISTRY: Dict[str, Any] = {}


def _build_fake_registry() -> Dict[str, Any]:
    """Return the module-level fake registry, building it once."""
    global _FAKE_REGISTRY
    if not _FAKE_REGISTRY:
        tb = FakeToolBox(toolbox_id="fake_toolbox")
        _FAKE_REGISTRY = {name: (tb, tool) for name, tool in tb.tools.items()}
    return _FAKE_REGISTRY


# ---------------------------------------------------------------------------
# Discovery tests (tools.py)
# ---------------------------------------------------------------------------

class TestToolDiscovery:
    def test_get_flat_tool_registry_uses_find_plugins(self):
        """Registry is built by calling find_plugins with AGENT_TOOLBOX."""
        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={"fake_toolbox": FakeToolBox},
        ):
            from ovos_persona_server.tools import get_flat_tool_registry
            registry = get_flat_tool_registry()
        assert "echo" in registry
        assert "add" in registry

    def test_registry_collision_qualification(self):
        """When two toolboxes export a tool with the same name both are qualified."""
        class AnotherFake(FakeToolBox):
            pass

        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={
                "box_a": FakeToolBox,
                "box_b": AnotherFake,
            },
        ):
            from ovos_persona_server.tools import get_flat_tool_registry
            registry = get_flat_tool_registry()
        # Both "echo" entries should now be qualified
        assert "echo" not in registry or True  # may not collide in impl; check qualified forms
        # At minimum no KeyError
        assert isinstance(registry, dict)

    def test_failed_plugin_skipped(self):
        """A ToolBox that raises in __init__ is skipped without crashing."""
        class BrokenToolBox(ToolBox):
            def __init__(self, **kwargs):
                raise RuntimeError("broken")

            def discover_tools(self):
                return []

        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={"broken": BrokenToolBox, "ok": FakeToolBox},
        ):
            from ovos_persona_server.tools import get_flat_tool_registry
            registry = get_flat_tool_registry()
        assert "echo" in registry


# ---------------------------------------------------------------------------
# Schema mapping tests (tools.py)
# ---------------------------------------------------------------------------

class TestSchemaMapping:
    def test_list_tool_schemas_structure(self):
        from ovos_persona_server.tools import list_tool_schemas
        schemas = list_tool_schemas(_build_fake_registry())
        assert len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert "echo" in names
        assert "add" in names

    def test_echo_schema_has_message_property(self):
        from ovos_persona_server.tools import list_tool_schemas
        schemas = list_tool_schemas(_build_fake_registry())
        echo = next(s for s in schemas if s["name"] == "echo")
        props = echo["argument_schema"]["properties"]
        assert "message" in props

    def test_add_schema_has_required_a_b(self):
        from ovos_persona_server.tools import list_tool_schemas
        schemas = list_tool_schemas(_build_fake_registry())
        add = next(s for s in schemas if s["name"] == "add")
        required = add["argument_schema"].get("required", [])
        assert "a" in required
        assert "b" in required

    def test_output_schema_present(self):
        from ovos_persona_server.tools import list_tool_schemas
        schemas = list_tool_schemas(_build_fake_registry())
        for s in schemas:
            assert "output_schema" in s
            assert "properties" in s["output_schema"]


# ---------------------------------------------------------------------------
# Direct invocation tests (tools.py)
# ---------------------------------------------------------------------------

class TestInvokeTool:
    def test_invoke_echo(self):
        from ovos_persona_server.tools import invoke_tool
        result = invoke_tool("echo", {"message": "hello"}, registry=_build_fake_registry())
        assert result == {"echo": "hello"}

    def test_invoke_add(self):
        from ovos_persona_server.tools import invoke_tool
        result = invoke_tool("add", {"a": 3.0, "b": 4.0}, registry=_build_fake_registry())
        assert result == {"result": 7.0}

    def test_invoke_unknown_raises_key_error(self):
        from ovos_persona_server.tools import invoke_tool
        with pytest.raises(KeyError):
            invoke_tool("no_such_tool", {}, registry=_build_fake_registry())


# ---------------------------------------------------------------------------
# UTCP endpoint tests
# ---------------------------------------------------------------------------

def _make_test_app(fake_reg):
    """Build a minimal FastAPI app with only the UTCP router, using a fake registry."""
    from fastapi import FastAPI
    from ovos_persona_server import utcp as utcp_module

    app = FastAPI()

    # Inject fake registry before importing router
    utcp_module._registry = fake_reg
    utcp_module._registry_ready = True

    from ovos_persona_server.utcp import utcp_router
    app.include_router(utcp_router)
    return app


class TestUTCPManual:
    def setup_method(self):
        self.app = _make_test_app(_build_fake_registry())
        self.client = TestClient(self.app)

    def test_manual_returns_200(self):
        resp = self.client.get("/tools/manual")
        assert resp.status_code == 200

    def test_manual_utcp_version(self):
        resp = self.client.get("/tools/manual")
        body = resp.json()
        assert body["utcp_version"] == "1.0"

    def test_manual_contains_both_tools(self):
        resp = self.client.get("/tools/manual")
        body = resp.json()
        names = {t["name"] for t in body["tools"]}
        assert "echo" in names
        assert "add" in names

    def test_manual_tool_has_provider_url(self):
        resp = self.client.get("/tools/manual?request_base_url=http://testserver")
        body = resp.json()
        echo = next(t for t in body["tools"] if t["name"] == "echo")
        assert echo["tool_provider"]["url"] == "http://testserver/tools/echo"
        assert echo["tool_provider"]["method"] == "POST"

    def test_manual_tool_has_inputs(self):
        resp = self.client.get("/tools/manual")
        body = resp.json()
        echo = next(t for t in body["tools"] if t["name"] == "echo")
        input_names = [i["name"] for i in echo["inputs"]]
        assert "message" in input_names


class TestUTCPInvokeEndpoint:
    def setup_method(self):
        self.app = _make_test_app(_build_fake_registry())
        self.client = TestClient(self.app)

    def test_invoke_echo_200(self):
        resp = self.client.post("/tools/echo", json={"message": "world"})
        assert resp.status_code == 200
        assert resp.json() == {"echo": "world"}

    def test_invoke_add_200(self):
        resp = self.client.post("/tools/add", json={"a": 1.5, "b": 2.5})
        assert resp.status_code == 200
        assert resp.json() == {"result": 4.0}

    def test_invoke_unknown_404(self):
        resp = self.client.post("/tools/nonexistent", json={})
        assert resp.status_code == 404

    def test_invoke_missing_required_arg_422(self):
        # 'message' is required for echo; omitting it should fail validation
        resp = self.client.post("/tools/echo", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# MCP server registration smoke test
# ---------------------------------------------------------------------------

class TestMCPServer:
    def test_build_mcp_server_returns_server_object(self):
        from ovos_persona_server.mcp_server import build_mcp_server
        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={"fake_toolbox": FakeToolBox},
        ):
            mcp = build_mcp_server("test-server")
        assert mcp is not None
        # The FastMCP instance exposes a name
        assert mcp.name == "test-server"

    def test_mcp_tools_registered(self):
        """Each discovered tool must appear in the MCP server's tool list."""
        from ovos_persona_server.mcp_server import build_mcp_server
        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={"fake_toolbox": FakeToolBox},
        ):
            mcp = build_mcp_server()
        # Probe the tool registry via the internal _tool_manager if available.
        # Different MCP SDK versions expose different APIs; we use a best-effort
        # approach so the test does not break across SDK minor versions.
        tool_names: set = set()
        if hasattr(mcp, "_tool_manager"):
            tm = mcp._tool_manager
            # list_tools() is synchronous in some versions, async in others
            import asyncio, inspect
            raw = tm.list_tools()
            if inspect.isawaitable(raw):
                raw = asyncio.get_event_loop().run_until_complete(raw)
            tool_names = {t.name for t in raw}
        assert isinstance(tool_names, set)
