"""MCP server: the tools it registers, the mount path, and the opt-in flag.

Installing the ``mcp`` extra must not, by itself, expose the ``/mcp``
endpoint — the flag is the agreed standard across the OVOS server family
(matching ``ovos-tts-server``). See also tests/e2e/test_e2e_mcp_utcp.py for
the real fastmcp transport behaviour.
"""
import json
from typing import Any, Dict, Generator, List
from unittest.mock import patch

import pytest
from pydantic import Field

from ovos_plugin_manager.templates.agent_tools import (
    AgentTool, ToolArguments, ToolBox, ToolOutput,
)

import ovos_persona_server
from ovos_persona_server import create_persona_app
from ovos_persona_server import persona as persona_mod
from tests.test_tools import FakeToolBox


class FakePersona:
    """Minimal stand-in for ovos_persona.Persona (mirrors test_multi_persona.py)."""

    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.memory = None
        self.seen: List = []

        class _Solvers:
            loaded_modules: dict = {}

        self.solvers = _Solvers()

    def chat(self, messages, sess=None, **kwargs) -> str:
        self.seen.append(messages)
        return f"hello from {self.name}"

    def stream(self, messages, sess=None, **kwargs) -> Generator[str, None, None]:
        self.seen.append(messages)
        yield f"hello from {self.name}"


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the process-wide persona registry out of other test modules."""
    saved = dict(persona_mod.personas)
    saved_default = persona_mod.default_persona
    yield
    persona_mod.personas.clear()
    persona_mod.personas.update(saved)
    persona_mod.default_persona = saved_default


@pytest.fixture()
def fake_persona_class(monkeypatch):
    """Make create_persona_app build FakePersona instead of a real Persona."""
    monkeypatch.setattr(ovos_persona_server, "Persona", FakePersona)
    return FakePersona


def _write_persona(tmp_path, name: str) -> str:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"name": name, "solvers": ["fake"]}))
    return str(path)


def _has_mcp_route(app) -> bool:
    return any(getattr(r, "path", "").startswith("/mcp") for r in app.routes)


def test_mcp_not_mounted_by_default(tmp_path, fake_persona_class):
    """No --mcp / enable_mcp -> no /mcp route, even if the extra is installed."""
    a = _write_persona(tmp_path, "alpha")
    app = create_persona_app([a])
    assert not _has_mcp_route(app)


def test_mcp_mounted_when_enabled(tmp_path, fake_persona_class):
    """enable_mcp=True -> /mcp route is present (requires the mcp extra)."""
    pytest.importorskip("fastmcp")
    a = _write_persona(tmp_path, "alpha")
    app = create_persona_app([a], enable_mcp=True)
    assert _has_mcp_route(app)


class _BlockFastMCPFinder:
    """A sys.meta_path finder that makes ``import fastmcp`` (and submodules)
    raise ImportError, simulating the ``mcp`` extra not being installed.

    Implements ``find_spec`` (not the legacy ``find_module``) — a finder that
    only defines ``find_module`` is silently skipped by the modern import
    system's ``PathFinder``/meta-path protocol, so the block would never
    actually fire and the test would pass for the wrong reason.
    """

    def find_spec(self, name, path, target=None):
        if name == "fastmcp" or name.startswith("fastmcp."):
            raise ImportError(f"No module named {name!r} (blocked for test)")
        return None


@pytest.fixture()
def _block_fastmcp():
    """Remove any cached fastmcp modules and block fresh imports of it."""
    import sys

    finder = _BlockFastMCPFinder()
    sys.meta_path.insert(0, finder)
    saved_modules = {
        name: mod for name, mod in sys.modules.items()
        if name == "fastmcp" or name.startswith("fastmcp.")
        or name == "ovos_persona_server.mcp_server"
    }
    for name in saved_modules:
        del sys.modules[name]
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name == "fastmcp" or name.startswith("fastmcp.") or name == "ovos_persona_server.mcp_server":
                del sys.modules[name]
        sys.modules.update(saved_modules)


def test_mcp_flag_without_extra_warns_not_crashes(tmp_path, fake_persona_class, _block_fastmcp, caplog):
    """enable_mcp=True without the mcp extra logs a warning naming the
    install command and still starts, rather than crashing or silently
    doing nothing (the failure mode the old bare ``except ImportError: pass``
    produced)."""
    a = _write_persona(tmp_path, "alpha")
    with caplog.at_level("WARNING"):
        app = create_persona_app([a], enable_mcp=True)
    assert not _has_mcp_route(app)
    warnings = [rec.message for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("mcp" in msg.lower() for msg in warnings), warnings
    assert any("install" in msg.lower() and "mcp]" in msg for msg in warnings), warnings


# ---------------------------------------------------------------------------
# MCP server registration smoke test
# ---------------------------------------------------------------------------

class TestMCPServer:
    def test_build_mcp_returns_server_object(self):
        from ovos_persona_server.mcp_server import build_mcp
        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={"fake_toolbox": FakeToolBox},
        ):
            mcp = build_mcp("test-server")
        assert mcp is not None
        # The FastMCP instance exposes a name
        assert mcp.name == "test-server"

    def test_mcp_tools_registered(self):
        """Each discovered tool must appear in the MCP server's tool list."""
        from ovos_persona_server.mcp_server import build_mcp
        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={"fake_toolbox": FakeToolBox},
        ):
            mcp = build_mcp()
        import asyncio
        tools = asyncio.run(mcp.list_tools())
        tool_names = {t.name for t in tools}
        assert tool_names == {"echo", "add"}


class TestMCPServerExtended:
    def test_mcp_handler_invokes_tool_and_returns_json(self):
        """Registered MCP handler wraps invoke_tool and returns JSON string."""
        from ovos_persona_server.mcp_server import build_mcp
        import asyncio
        import json as _json
        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={"fake_toolbox": FakeToolBox},
        ):
            mcp = build_mcp()
        result = asyncio.run(mcp.call_tool("echo", {"message": "hello MCP"}))
        text = result.content[0].text
        assert _json.loads(text) == {"echo": "hello MCP"}

    def test_mcp_tool_failure_is_an_mcp_tool_error(self):
        """A failing tool answers with isError true and the reason as text.

        MCP reports tool execution errors "in tool results with isError:
        true", so a client can tell a failure from a result."""
        from ovos_persona_server.mcp_server import build_mcp

        class BoomArgs(ToolArguments):
            x: str = Field(default="x")

        class BoomOutput(ToolOutput):
            y: str = Field(default="y")

        class BoomBox(ToolBox):
            def __init__(self, config: Dict[str, Any] = None, bus: Any = None) -> None:
                super().__init__(toolbox_id="boom_box", config=config, bus=bus)

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
            mcp = build_mcp()

        import asyncio
        from fastmcp import Client

        async def call():
            async with Client(mcp) as client:
                return await client.call_tool("errortool", {"x": "test"},
                                              raise_on_error=False)

        result = asyncio.run(call())
        assert result.is_error is True
        assert "Tool execution failed for 'errortool'" in result.content[0].text

    def test_mcp_build_with_no_plugins_succeeds(self):
        """build_mcp with empty plugin list succeeds and logs info."""
        from ovos_persona_server.mcp_server import build_mcp
        with patch(
            "ovos_persona_server.tools.find_plugins",
            return_value={},
        ):
            mcp = build_mcp("empty-server")
        assert mcp.name == "empty-server"


class TestLayoutNames:
    def test_server_name_is_ovos_persona(self):
        from ovos_persona_server.mcp_server import build_mcp
        with patch("ovos_persona_server.tools.find_plugins", return_value={}):
            assert build_mcp().name == "ovos-persona"

    def test_server_info_version_is_the_package_version(self):
        """MCP serverInfo.version names this package, not fastmcp."""
        import asyncio
        from fastmcp import Client
        from ovos_persona_server.mcp_server import build_mcp
        from ovos_persona_server.version import __version__

        async def server_info():
            with patch("ovos_persona_server.tools.find_plugins", return_value={}):
                mcp = build_mcp()
            async with Client(mcp) as client:
                return client.server_info

        info = asyncio.run(server_info())
        assert (info.name, info.version) == ("ovos-persona", __version__)

    def test_mount_mcp_serves_at_exactly_the_path(self):
        from fastapi import FastAPI
        from ovos_persona_server.mcp_server import mount_mcp
        app = FastAPI()
        with patch("ovos_persona_server.tools.find_plugins", return_value={}):
            mount_mcp(app)
        assert [r.path for r in app.routes if r.path.startswith("/mcp")] == ["/mcp"]

    @pytest.mark.parametrize("old,new", [("build_mcp_server", "build_mcp"),
                                         ("mount_mcp_on_app", "mount_mcp")])
    def test_old_names_are_deprecated_aliases(self, old, new):
        from fastapi import FastAPI
        from ovos_persona_server import mcp_server
        from ovos_persona_server.version import VERSION_MAJOR, VERSION_MINOR
        args = (FastAPI(),) if old == "mount_mcp_on_app" else ()
        with patch("ovos_persona_server.tools.find_plugins", return_value={}), \
                pytest.warns(DeprecationWarning,
                             match=f"use {new}; it will be removed in "
                                   f"{VERSION_MAJOR}.{VERSION_MINOR + 1}.0"):
            getattr(mcp_server, old)(*args)


_INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
               "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                          "clientInfo": {"name": "test", "version": "0"}}}


@pytest.mark.parametrize("path", ["/mcp", "/mcp/"])
def test_post_mcp_reaches_the_transport_without_a_redirect(path):
    """A 307 from ``/mcp`` to ``/mcp/`` names ``http://`` behind a TLS proxy
    that uvicorn does not trust, and MCP clients refuse it."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ovos_persona_server.mcp_server import mount_mcp

    app = FastAPI()
    with patch("ovos_persona_server.tools.find_plugins", return_value={}):
        mount_mcp(app)
    with TestClient(app, follow_redirects=False) as client:
        resp = client.post(path, json=_INITIALIZE, headers={
            "Accept": "application/json, text/event-stream"})
    assert resp.status_code == 200, (resp.status_code, resp.headers.get("location"))
    assert "ovos-persona" in resp.text


def test_missing_extra_warning_shows_at_the_default_log_level(tmp_path):
    """Run in a fresh interpreter with no logging set up by the test runner,
    so a package that lowered the global log level would hide the warning."""
    import subprocess
    import sys
    import textwrap

    persona = _write_persona(tmp_path, "alpha")
    code = textwrap.dedent(f"""
        import sys
        sys.modules["fastmcp"] = None
        import ovos_persona_server
        from types import SimpleNamespace
        ovos_persona_server.Persona = lambda name, config: SimpleNamespace(
            name=name, config=config)
        app = ovos_persona_server.create_persona_app([{persona!r}], enable_mcp=True)
        assert not any(getattr(r, "path", "") == "/mcp" for r in app.routes)
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert 'pip install "ovos-persona-server[mcp]"' in proc.stderr


def test_stdio_entry_point_writes_only_json_rpc_to_stdout(tmp_path):
    """Run the installed console script's target. Its first stdout line must
    be the JSON-RPC answer to initialize; a log line there breaks the MCP
    stream for the client."""
    import os
    import subprocess
    import sys
    from importlib.metadata import entry_points
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    (script,) = entry_points().select(group="console_scripts",
                                      name="ovos-persona-tools-mcp")
    module, func = script.module, script.attr
    initialize = json.dumps(_INITIALIZE) + "\n"
    env = dict(os.environ, PYTHONPATH=str(repo), PERSONA_PATH="")
    proc = subprocess.run(
        [sys.executable, "-c", f"from {module} import {func}; {func}()"],
        input=initialize, capture_output=True, text=True, timeout=120,
        env=env, cwd=tmp_path)
    first = proc.stdout.splitlines()[0] if proc.stdout else ""
    reply = json.loads(first)
    assert (reply["jsonrpc"], reply["id"]) == ("2.0", 1)
    assert reply["result"]["serverInfo"]["name"] == "ovos-persona"
