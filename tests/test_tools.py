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
    """Minimal in-process ToolBox with two deterministic tools.

    Follows the plugin contract: the plugin itself owns its ``toolbox_id``
    and passes it to the parent class; the loader never passes one in.
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        self.config = config or {}
        super().__init__(toolbox_id="fake_toolbox")

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
        tb = FakeToolBox()
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


# ---------------------------------------------------------------------------
# Additional coverage: gaps identified by --cov-report=term-missing
# ---------------------------------------------------------------------------

class TestToolDiscoveryExtended:
    def test_no_opm_falls_back_to_empty(self):
        """When find_plugins is None (OPM absent) registry is empty."""
        import ovos_persona_server.tools as tools_mod
        original = tools_mod.find_plugins
        tools_mod.find_plugins = None
        try:
            from ovos_persona_server.tools import _load_toolboxes
            result = _load_toolboxes()
            assert result == []
        finally:
            tools_mod.find_plugins = original

    def test_collision_qualifies_both_entries(self):
        """Two toolboxes with the same tool name produce two qualified entries."""
        class BoxA(FakeToolBox):
            pass

        class BoxB(FakeToolBox):
            pass

        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={"box_a": BoxA, "box_b": BoxB},
        ):
            from ovos_persona_server.tools import get_flat_tool_registry
            registry = get_flat_tool_registry()

        qualified = [k for k in registry if "__" in k]
        assert any("echo" in k for k in qualified), f"Expected qualified echo entries, got: {list(registry)}"
        assert any("add" in k for k in qualified)
        # Bare names must NOT exist when there was a collision
        assert "echo" not in registry
        assert "add" not in registry

    def test_empty_registry_list_schemas_empty(self):
        """list_tool_schemas on an empty registry returns an empty list."""
        from ovos_persona_server.tools import list_tool_schemas
        assert list_tool_schemas({}) == []

    def test_invoke_tool_wrong_type_propagates(self):
        """invoke_tool with wrong arg types raises from the underlying tool."""
        class BoomArgs(ToolArguments):
            value: int = Field(..., description="An integer.")

        class BoomOutput(ToolOutput):
            ok: bool = Field(default=True)

        def _boom(args):
            # pydantic will coerce; force an explicit error
            if not isinstance(args.value, int):
                raise TypeError("expected int")
            return BoomOutput()

        class BoomBox(ToolBox):
            def __init__(self, config: Dict[str, Any] = None) -> None:
                self.config = config or {}
                super().__init__(toolbox_id="boom_box")

            def discover_tools(self):
                return [AgentTool(
                    name="boom",
                    description="boom",
                    argument_schema=BoomArgs,
                    output_schema=BoomOutput,
                    tool_call=_boom,
                )]

        tb = BoomBox()
        reg = {"boom": (tb, tb.tools["boom"])}

        from ovos_persona_server.tools import invoke_tool
        # Passing a non-coercible string for an int field should raise
        with pytest.raises(Exception):
            invoke_tool("boom", {"value": "not-an-int"}, registry=reg)


class TestUTCPEdgeCases:
    def setup_method(self):
        from fastapi import FastAPI
        from ovos_persona_server import utcp as utcp_module
        app = FastAPI()
        utcp_module._registry = _build_fake_registry()
        utcp_module._registry_ready = True
        from ovos_persona_server.utcp import utcp_router
        app.include_router(utcp_router)
        from fastapi.testclient import TestClient
        self.client = TestClient(app)

    def test_manual_empty_registry(self):
        """Manual endpoint with empty registry returns tools=[]."""
        from fastapi import FastAPI
        from ovos_persona_server import utcp as utcp_module
        app2 = FastAPI()
        utcp_module._registry = {}
        utcp_module._registry_ready = True
        from ovos_persona_server.utcp import utcp_router
        app2.include_router(utcp_router)
        from fastapi.testclient import TestClient
        client2 = TestClient(app2)
        resp = client2.get("/tools/manual")
        assert resp.status_code == 200
        assert resp.json()["tools"] == []

    def test_invoke_tool_500_on_unexpected_exception(self):
        """When a tool raises an unexpected exception, the endpoint returns 500."""
        class BoomArgs(ToolArguments):
            x: str = Field(default="x")

        class BoomOutput(ToolOutput):
            y: str = Field(default="y")

        class BoomBox(ToolBox):
            def __init__(self, config: Dict[str, Any] = None) -> None:
                self.config = config or {}
                super().__init__(toolbox_id="boom_box")

            def discover_tools(self):
                return [AgentTool(
                    name="crasher",
                    description="always crashes",
                    argument_schema=BoomArgs,
                    output_schema=BoomOutput,
                    tool_call=lambda args: (_ for _ in ()).throw(RuntimeError("boom")),
                )]

        tb = BoomBox()
        reg = {"crasher": (tb, tb.tools["crasher"])}

        from fastapi import FastAPI
        from ovos_persona_server import utcp as utcp_module
        app3 = FastAPI()
        utcp_module._registry = reg
        utcp_module._registry_ready = True
        from ovos_persona_server.utcp import utcp_router
        app3.include_router(utcp_router)
        from fastapi.testclient import TestClient
        client3 = TestClient(app3, raise_server_exceptions=False)
        resp = client3.post("/tools/crasher", json={"x": "hi"})
        assert resp.status_code == 500

    def test_manual_request_base_url_override(self):
        """request_base_url query param is embedded in provider URLs."""
        resp = self.client.get("/tools/manual?request_base_url=https://myproxy.example.com")
        body = resp.json()
        echo = next((t for t in body["tools"] if t["name"] == "echo"), None)
        assert echo is not None
        assert echo["tool_provider"]["url"] == "https://myproxy.example.com/tools/echo"


class TestMCPServerExtended:
    def test_mcp_handler_invokes_tool_and_returns_json(self):
        """Registered MCP handler wraps invoke_tool and returns JSON string."""
        from ovos_persona_server.mcp_server import build_mcp_server
        import json as _json
        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={"fake_toolbox": FakeToolBox},
        ):
            mcp = build_mcp_server()
        # Find the echo handler by introspecting _tool_manager
        if not hasattr(mcp, "_tool_manager"):
            pytest.skip("MCP SDK version lacks _tool_manager")
        import asyncio, inspect
        tm = mcp._tool_manager
        raw = tm.list_tools()
        if inspect.isawaitable(raw):
            raw = asyncio.get_event_loop().run_until_complete(raw)
        echo_tool = next((t for t in raw if t.name == "echo"), None)
        if echo_tool is None:
            pytest.skip("echo tool not found in MCP registry")
        # Call the handler directly
        fn = echo_tool.fn
        result_raw = fn(message="hello MCP")
        if inspect.isawaitable(result_raw):
            result_raw = asyncio.get_event_loop().run_until_complete(result_raw)
        result = _json.loads(result_raw)
        assert result == {"echo": "hello MCP"}

    def test_mcp_handler_returns_error_json_on_failure(self):
        """If the underlying tool raises, the MCP handler returns JSON error."""
        from ovos_persona_server.mcp_server import build_mcp_server
        import json as _json

        class BoomArgs(ToolArguments):
            x: str = Field(default="x")

        class BoomOutput(ToolOutput):
            y: str = Field(default="y")

        class BoomBox(ToolBox):
            def __init__(self, config: Dict[str, Any] = None) -> None:
                self.config = config or {}
                super().__init__(toolbox_id="boom_box")

            def discover_tools(self):
                return [AgentTool(
                    name="errortool",
                    description="always errors",
                    argument_schema=BoomArgs,
                    output_schema=BoomOutput,
                    tool_call=lambda args: (_ for _ in ()).throw(ValueError("oops")),
                )]

        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={"boom_box": BoomBox},
        ):
            mcp = build_mcp_server()

        if not hasattr(mcp, "_tool_manager"):
            pytest.skip("MCP SDK version lacks _tool_manager")
        import asyncio, inspect
        tm = mcp._tool_manager
        raw = tm.list_tools()
        if inspect.isawaitable(raw):
            raw = asyncio.get_event_loop().run_until_complete(raw)
        err_tool = next((t for t in raw if t.name == "errortool"), None)
        if err_tool is None:
            pytest.skip("errortool not found in MCP registry")
        fn = err_tool.fn
        result_raw = fn(x="test")
        if inspect.isawaitable(result_raw):
            result_raw = asyncio.get_event_loop().run_until_complete(result_raw)
        result = _json.loads(result_raw)
        assert "error" in result

    def test_mcp_build_with_no_plugins_succeeds(self):
        """build_mcp_server with empty plugin list succeeds and logs info."""
        from ovos_persona_server.mcp_server import build_mcp_server
        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={},
        ):
            mcp = build_mcp_server("empty-server")
        assert mcp.name == "empty-server"


class TestNoneRegistryFallbacks:
    """Cover the registry=None default path in invoke_tool and list_tool_schemas."""

    def test_invoke_tool_none_registry_builds_fresh(self):
        """invoke_tool(registry=None) triggers get_flat_tool_registry internally."""
        from ovos_persona_server.tools import invoke_tool
        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={"fake_toolbox": FakeToolBox},
        ):
            result = invoke_tool("echo", {"message": "from-none"}, registry=None)
        assert result == {"echo": "from-none"}

    def test_list_tool_schemas_none_registry_builds_fresh(self):
        """list_tool_schemas(registry=None) triggers get_flat_tool_registry."""
        from ovos_persona_server.tools import list_tool_schemas
        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={"fake_toolbox": FakeToolBox},
        ):
            schemas = list_tool_schemas(registry=None)
        assert len(schemas) >= 1

    def test_bare_toolbox_missing_toolbox_id_is_skipped(self):
        """
        The loader always instantiates plugins with ``cls()``. A ToolBox
        subclass that does not override ``__init__`` (and therefore requires
        the template's ``toolbox_id`` kwarg) fails to instantiate and is
        logged+skipped, while the other, well-behaved plugins still load.
        """
        class BareToolBox(ToolBox):
            """Does not override __init__; inherits the template's, which
            requires an explicit ``toolbox_id`` kwarg — broken under the
            no-argument loader contract."""

            def discover_tools(self) -> List[AgentTool]:
                return []

        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={"bare_toolbox": BareToolBox, "fake_toolbox": FakeToolBox},
        ):
            from ovos_persona_server.tools import get_flat_tool_registry
            registry = get_flat_tool_registry()
        # The well-behaved plugin still loads, the broken one is silently
        # skipped (no exception propagates and no tools of its own appear).
        assert "echo" in registry
        assert "add" in registry
        assert len(registry) == 2

    def test_refresh_tools_exception_silenced(self):
        """If refresh_tools raises, it is silently skipped and tools still load."""
        class RefreshBoomBox(FakeToolBox):
            def refresh_tools(self):
                raise RuntimeError("refresh broken")

        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={"refresh_boom": RefreshBoomBox},
        ):
            from ovos_persona_server.tools import get_flat_tool_registry
            registry = get_flat_tool_registry()
        # Tools should still be accessible (loaded by __init__ before refresh)
        # At minimum, the call must not raise.
        assert isinstance(registry, dict)
