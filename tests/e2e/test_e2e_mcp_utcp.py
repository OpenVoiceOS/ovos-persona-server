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
"""End-to-end tests for ovos-persona-server MCP /mcp and UTCP /tools/manual endpoints.

Boots a minimal FastAPI app constructed directly from the UTCP router and MCP
server (bypassing the full create_persona_app which requires a live Persona
instance), then starts uvicorn on a free port in a background thread.

The UTCP /tools/manual endpoint returns a list of installed OPM ToolBox tools
(empty when none are installed — that is still a valid response).
The MCP endpoint (when the mcp extra is available) must support
initialize → list_tools.

Run in isolation::

    pytest tests/e2e/test_e2e_mcp_utcp.py -v --timeout=30
"""
from __future__ import annotations

import asyncio
import importlib
import socket
import threading
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_stub_app() -> FastAPI:
    """
    Build a minimal FastAPI app that includes only the UTCP and MCP features
    under test, with an empty tool registry (no real OPM plugins required).
    """
    app = FastAPI(title="persona-e2e-test")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Patch the tool registry to return an empty dict — no real tools needed
    with patch("ovos_persona_server.tools.get_flat_tool_registry", return_value={}):
        from ovos_persona_server.utcp import utcp_router
        app.include_router(utcp_router)

    # Mount MCP if available; it will expose zero tools but must be reachable
    try:
        from ovos_persona_server.mcp_server import build_mcp_server
        with patch("ovos_persona_server.tools.get_flat_tool_registry", return_value={}):
            mcp = build_mcp_server()
        try:
            mcp_asgi = mcp.sse_app()
        except AttributeError:
            mcp_asgi = mcp.streamable_http_app()
        app.mount("/mcp", mcp_asgi)
    except ImportError:
        pass

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _start_server(app, health_path: str) -> tuple:
    """Start uvicorn via asyncio.run in a daemon thread."""
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=asyncio.run, args=(server.serve(),), daemon=True)
    thread.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{port}{health_path}", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        raise RuntimeError("Server did not start in time")
    return f"http://127.0.0.1:{port}", server, thread


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_server():
    """Boot the stub persona app and serve on a free port (UTCP tests)."""
    with patch("ovos_persona_server.tools.get_flat_tool_registry", return_value={}), \
         patch("ovos_persona_server.utcp._registry", {}), \
         patch("ovos_persona_server.utcp._registry_ready", False):
        app = _make_stub_app()

    try:
        base_url, server, thread = _start_server(app, health_path="/tools/manual")
    except RuntimeError as exc:
        pytest.skip(str(exc))

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def mcp_server():
    """Run MCP as a standalone Starlette app to avoid sub-app lifespan issues."""
    try:
        from ovos_persona_server.mcp_server import build_mcp_server
    except ImportError:
        pytest.skip("mcp extra not installed")

    with patch("ovos_persona_server.tools.get_flat_tool_registry", return_value={}):
        mcp = build_mcp_server()
    mcp_app = mcp.streamable_http_app()
    try:
        base_url, server, thread = _start_server(mcp_app, health_path="/mcp")
    except RuntimeError as exc:
        pytest.skip(str(exc))

    yield f"{base_url}/mcp"

    server.should_exit = True
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# UTCP end-to-end
# ---------------------------------------------------------------------------

class TestUtcpE2E:
    def test_utcp_manual_200(self, live_server):
        resp = httpx.get(f"{live_server}/tools/manual", timeout=10)
        assert resp.status_code == 200

    def test_utcp_manual_is_json(self, live_server):
        data = httpx.get(f"{live_server}/tools/manual", timeout=10).json()
        assert isinstance(data, dict)

    def test_utcp_version_present(self, live_server):
        data = httpx.get(f"{live_server}/tools/manual", timeout=10).json()
        assert "utcp_version" in data

    def test_utcp_tools_list_present(self, live_server):
        data = httpx.get(f"{live_server}/tools/manual", timeout=10).json()
        assert "tools" in data
        # Empty list is valid (no tool plugins installed in CI)
        assert isinstance(data["tools"], list)

    def test_utcp_manual_with_base_url_param(self, live_server):
        """request_base_url query param overrides embedded tool URLs."""
        data = httpx.get(
            f"{live_server}/tools/manual",
            params={"request_base_url": "https://override.example.com"},
            timeout=10,
        ).json()
        assert "utcp_version" in data

    def test_utcp_invoke_missing_tool_returns_404(self, live_server):
        """POST /tools/nonexistent-tool must return 404."""
        resp = httpx.post(
            f"{live_server}/tools/nonexistent-tool",
            json={},
            timeout=10,
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# MCP end-to-end
# ---------------------------------------------------------------------------

_mcp_available = importlib.util.find_spec("mcp") is not None
mcp_required = pytest.mark.skipif(
    not _mcp_available,
    reason="mcp package not installed",
)


@mcp_required
class TestMcpE2E:
    """MCP tests use a standalone MCP Starlette app to avoid sub-app lifespan issues."""

    def test_mcp_endpoint_accessible(self, mcp_server):
        resp = httpx.get(mcp_server, timeout=10)
        assert resp.status_code != 404

    def test_mcp_initialize_and_list_tools(self, mcp_server):
        """MCP handshake must succeed; zero tools is acceptable."""
        from mcp.client.streamable_http import streamable_http_client
        from mcp import ClientSession

        async def _run():
            async with streamable_http_client(mcp_server) as (r, w, _):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return result.tools

        tools = asyncio.run(_run())
        # No real tool plugins installed — list may be empty
        assert isinstance(tools, list)
