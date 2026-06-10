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
MCP server that exposes installed OPM tool plugins as MCP tools.

Each ``ToolBox`` plugin's tools are discovered at startup and registered with
the ``mcp`` Python SDK.  The server is mounted on the FastAPI app at
``/mcp`` (SSE transport) so it shares the existing Uvicorn process.

Usage (standalone, for an MCP client that supports the stdio transport)::

    ovos-persona-tools-mcp

Usage (as part of the persona server, automatic when ``[mcp]`` extras are
installed)::

    ovos-persona-server --persona mybot.json

MCP clients connecting via SSE point at::

    http://<host>:<port>/mcp/sse
"""

import json
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP
from ovos_utils.log import LOG

from ovos_persona_server.tools import get_flat_tool_registry, invoke_tool, list_tool_schemas


def build_mcp_server(name: str = "ovos-persona-tools") -> FastMCP:
    """
    Construct and return a :class:`~mcp.server.fastmcp.FastMCP` instance with
    all installed OPM tool plugins registered as MCP tools.

    The registry is built once at call-time; the server is stateless
    afterwards (tools are effectively frozen until the process restarts).

    Args:
        name: Human-readable server name reported to MCP clients.

    Returns:
        Configured ``FastMCP`` server ready to be mounted or run standalone.
    """
    mcp = FastMCP(name)

    registry = get_flat_tool_registry()
    schemas = list_tool_schemas(registry)

    if not schemas:
        LOG.info("MCP server: no OPM tool plugins found; server will have no tools")

    for tool_def in schemas:
        tool_name: str = tool_def["name"]
        tool_description: str = tool_def["description"]
        arg_schema: Dict[str, Any] = tool_def["argument_schema"]

        # Build a dynamic wrapper that closes over the tool name and registry.
        # The MCP SDK expects the function signature to carry kwarg annotations
        # so that it can generate its own JSON Schema; we supply a pre-built one.
        def _make_handler(tname: str):
            def handler(**kwargs: Any) -> str:
                """Invoke the OPM tool plugin and return JSON-serialised output."""
                try:
                    result = invoke_tool(tname, kwargs, registry=registry)
                    return json.dumps(result)
                except Exception as exc:
                    LOG.error("MCP tool %s failed: %s", tname, exc)
                    return json.dumps({"error": str(exc)})
            handler.__name__ = tname
            handler.__doc__ = tool_description
            return handler

        handler_fn = _make_handler(tool_name)

        # Register with the MCP SDK.  We pass the JSON Schema directly so the
        # SDK does not try to introspect the dynamic handler's signature.
        mcp.tool(
            name=tool_name,
            description=tool_description,
        )(handler_fn)

        LOG.debug("MCP: registered tool %s", tool_name)

    return mcp



def mount_mcp_on_app(app, path="/mcp", name="ovos-persona-tools"):
    """Mount MCP streamable-HTTP transport at path with lifespan chaining."""
    from contextlib import asynccontextmanager
    mcp = build_mcp_server(name=name)
    mcp.settings.streamable_http_path = "/"
    app.mount(path, mcp.streamable_http_app())
    _orig = app.router.lifespan_context
    @asynccontextmanager
    async def _wrap(h):
        async with _orig(h):
            async with mcp.session_manager.run():
                yield
    app.router.lifespan_context = _wrap
    LOG.info("MCP server mounted at %s (streamable-HTTP)", path)


def _run_stdio() -> None:
    """Entry point for the ``ovos-persona-tools-mcp`` console script (stdio)."""
    server = build_mcp_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    _run_stdio()
