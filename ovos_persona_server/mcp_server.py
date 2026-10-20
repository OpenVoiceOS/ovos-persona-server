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
the ``fastmcp`` package. :func:`mount_mcp` serves them on the FastAPI app at
``/mcp`` (streamable-HTTP transport) so they share the Uvicorn process.

Usage (standalone, for an MCP client that supports the stdio transport)::

    ovos-persona-tools-mcp

Usage (as part of the persona server, opt-in via ``--mcp`` even when the
``[mcp]`` extra is installed)::

    ovos-persona-server --persona mybot.json --mcp

MCP clients connect to::

    http://<host>:<port>/mcp
"""

import json
import warnings
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools.function_tool import FunctionTool
from ovos_utils.log import LOG

from ovos_persona_server.tools import get_flat_tool_registry, invoke_tool, list_tool_schemas
from ovos_persona_server.version import VERSION_MAJOR, VERSION_MINOR, __version__

__all__ = ["build_mcp", "mount_mcp"]


def build_mcp(name: str = "ovos-persona") -> FastMCP:
    """
    Construct and return a :class:`~fastmcp.FastMCP` instance with all
    installed OPM tool plugins registered as MCP tools.

    The registry is built once at call-time; the server is stateless
    afterwards (tools are effectively frozen until the process restarts).

    Args:
        name: Human-readable server name reported to MCP clients.

    Returns:
        Configured ``FastMCP`` server ready to be mounted or run standalone.
    """
    mcp = FastMCP(name, version=__version__)

    registry = get_flat_tool_registry()
    schemas = list_tool_schemas(registry)

    if not schemas:
        LOG.info("MCP server: no OPM tool plugins found; server will have no tools")

    for tool_def in schemas:
        tool_name: str = tool_def["name"]
        tool_description: str = tool_def["description"]
        arg_schema: Dict[str, Any] = tool_def["argument_schema"]

        # Build a dynamic wrapper that closes over the tool name and registry.
        # fastmcp's ``@mcp.tool`` decorator introspects the handler's
        # signature to build its JSON Schema and rejects ``**kwargs``
        # handlers outright, so we construct a ``FunctionTool`` directly and
        # hand it the pre-built JSON Schema instead of letting fastmcp infer
        # one from the (dynamic, kwargs-only) handler signature.
        def _make_handler(tname: str):
            def handler(**kwargs: Any) -> str:
                """Invoke the OPM tool plugin and return JSON-serialised output."""
                try:
                    result = invoke_tool(tname, kwargs, registry=registry)
                except Exception as exc:
                    LOG.error("MCP tool %s failed: %s", tname, exc)
                    # MCP reports a tool execution error as a result with
                    # isError true; ToolError carries the reason through.
                    raise ToolError(str(exc)) from exc
                return json.dumps(result)
            handler.__name__ = tname
            handler.__doc__ = tool_description
            return handler

        handler_fn = _make_handler(tool_name)

        mcp.add_tool(
            FunctionTool(
                name=tool_name,
                description=tool_description,
                parameters=arg_schema,
                fn=handler_fn,
            )
        )

        LOG.debug("MCP: registered tool %s", tool_name)

    return mcp



class _ExactPath:
    """ASGI middleware that hands ``path`` to the mount as ``path/``.

    A Starlette ``Mount(path)`` matches only ``path/...`` and answers
    ``POST path`` with a 307 to ``path/``. Behind a TLS proxy that uvicorn
    does not trust, that redirect names ``http://``, and MCP clients refuse
    it. Rewriting the path before routing serves both spellings directly.
    """

    def __init__(self, app, path: str) -> None:
        self.app = app
        self.path = path

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and scope["path"] == self.path:
            scope = dict(scope, path=self.path + "/",
                         raw_path=(self.path + "/").encode())
        await self.app(scope, receive, send)


def mount_mcp(app, path: str = "/mcp", name: str = "ovos-persona") -> None:
    """Mount the MCP streamable-HTTP transport on *app* at exactly *path*.

    The sub-app is built with ``path="/"`` so the endpoint is *path* and not
    *path*/mcp, and its lifespan, which runs the MCP session manager, is
    chained into the host app's lifespan because Starlette does not start a
    mounted app's lifespan. ``POST path`` and ``POST path/`` both reach the
    transport with no redirect.
    """
    mcp_app = build_mcp(name=name).http_app(path="/", transport="streamable-http")
    app.mount(path, mcp_app)
    app.add_middleware(_ExactPath, path=path)
    host_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _lifespan_with_mcp(host_app):
        async with host_lifespan(host_app):
            async with mcp_app.lifespan(mcp_app):
                yield

    app.router.lifespan_context = _lifespan_with_mcp
    LOG.info("MCP server mounted at %s (streamable-HTTP)", path)


def _deprecated(old: str, new: str) -> None:
    warnings.warn(
        f"{old} is deprecated, use {new}; "
        f"it will be removed in {VERSION_MAJOR}.{VERSION_MINOR + 1}.0",
        DeprecationWarning, stacklevel=3)


def build_mcp_server(name: str = "ovos-persona") -> FastMCP:
    """Deprecated: use :func:`build_mcp`."""
    _deprecated("build_mcp_server", "build_mcp")
    return build_mcp(name=name)


def mount_mcp_on_app(app, path: str = "/mcp", name: str = "ovos-persona") -> None:
    """Deprecated: use :func:`mount_mcp`."""
    _deprecated("mount_mcp_on_app", "mount_mcp")
    mount_mcp(app, path=path, name=name)


def _run_stdio() -> None:
    """Entry point for the ``ovos-persona-tools-mcp`` console script (stdio)."""
    build_mcp().run(transport="stdio")


if __name__ == "__main__":
    _run_stdio()
