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
UTCP (Universal Tool Calling Protocol) endpoints.

``GET /utcp``
    Returns a UTCP 1.0 manual describing all installed OPM tool plugins as
    HTTP-callable tools. Clients parse this once and know how to call each
    tool. ``GET /tools/manual`` serves the same manual.

``POST /tools/{name}``
    Generic invocation endpoint. The request body is a JSON object whose
    fields are the tool's keyword arguments. Returns the tool's output as
    JSON.

UTCP manual format reference:
    https://www.utcp.io/docs/for-tool-providers
"""

import warnings
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from ovos_persona_server.tools import get_flat_tool_registry, invoke_tool, list_tool_schemas
from ovos_persona_server.version import VERSION_MAJOR, VERSION_MINOR, __version__

__all__ = ["build_utcp_manual", "make_utcp_router", "MANUAL_VERSION", "UTCP_VERSION"]

# The UTCP specification the manual follows, and the version of the tools
# it describes, which is the package version.
UTCP_VERSION = "1.0.0"
MANUAL_VERSION = __version__

# ---------------------------------------------------------------------------
# Registry is initialised lazily on first request so the server can start
# even if no tool plugins are installed yet.
# ---------------------------------------------------------------------------
_registry: Dict[str, Any] = {}
_registry_ready: bool = False


def _ensure_registry() -> Dict[str, Any]:
    global _registry, _registry_ready
    if not _registry_ready:
        _registry = get_flat_tool_registry()
        _registry_ready = True
    return _registry


def _utcp_tool_entry(base_url: str, tool_def: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a tool descriptor dict into a UTCP 1.0 tool entry.

    ``POST /tools/{name}`` reads the tool arguments as one JSON object, so
    the manual declares a single ``body`` input that holds them and maps it
    to the request body with ``body_field``. This is the shape the UTCP HTTP
    client gives every JSON request body.

    Args:
        base_url: The server's base URL (e.g. ``http://localhost:8337``).
        tool_def: Tool descriptor from :func:`list_tool_schemas`.

    Returns:
        UTCP-formatted tool dict.
    """
    return {
        "name": tool_def["name"],
        "description": tool_def["description"],
        "inputs": {
            "type": "object",
            "properties": {"body": tool_def.get("argument_schema", {})},
            "required": ["body"],
        },
        "outputs": tool_def.get("output_schema", {}),
        "tags": ["ovos", "persona-tool"],
        "tool_call_template": {
            "call_template_type": "http",
            "name": "ovos_persona_tools",
            "http_method": "POST",
            "url": f"{base_url}/tools/{tool_def['name']}",
            "content_type": "application/json",
            "body_field": "body",
        },
    }


def build_utcp_manual(base_url: str) -> Dict[str, Any]:
    """Return the UTCP 1.0 manual for every installed OPM tool.

    Args:
        base_url: The server's public base URL, e.g. ``http://localhost:8337``.
            A trailing slash is stripped.

    Returns:
        A dict conforming to the UTCP ``UtcpManual`` schema.
    """
    base_url = base_url.rstrip("/")
    schemas = list_tool_schemas(_ensure_registry())
    return {
        "utcp_version": UTCP_VERSION,
        "manual_version": MANUAL_VERSION,
        "tools": [_utcp_tool_entry(base_url, td) for td in schemas],
    }


def make_utcp_router() -> APIRouter:
    """Return a router serving ``GET /utcp``, ``GET /tools/manual`` and
    ``POST /tools/{name}``.

    The manual's base URL is read from each request, so the tool URLs name
    the host and port the client used.
    """
    router = APIRouter(tags=["utcp"])

    async def utcp_manual(request: Request,
                          request_base_url: Optional[str] = None) -> JSONResponse:
        """
        Return a UTCP 1.0 manual describing all installed OPM tool plugins.

        The tool URLs use the base URL of the incoming request. The
        ``request_base_url`` query parameter replaces it, for a proxy that
        does not forward the public host.

        Example::

            GET /utcp?request_base_url=https://myserver.example.com
        """
        return JSONResponse(content=build_utcp_manual(
            request_base_url or str(request.base_url)))

    router.add_api_route("/utcp", utcp_manual, methods=["GET"],
                         summary="UTCP manual — agent tool discovery",
                         response_class=JSONResponse)
    router.add_api_route("/tools/manual", utcp_manual, methods=["GET"],
                         summary="UTCP manual — same document as GET /utcp",
                         response_class=JSONResponse)

    @router.post(
        "/tools/{name}",
        summary="Invoke an OPM tool plugin by name",
        response_class=JSONResponse,
    )
    async def invoke_tool_endpoint(name: str, body: Dict[str, Any] = None) -> JSONResponse:
        """
        Invoke a named OPM tool plugin and return its output as JSON.

        The request body must be a JSON object whose keys match the tool's
        declared input parameters (see ``GET /utcp`` for schemas).

        Args:
            name: The tool name as listed in the UTCP manual.
            body: JSON request body (tool kwargs).

        Returns:
            JSON object matching the tool's ``output_schema``.

        Raises:
            404: If no tool with that name is registered.
            422: If the tool's input validation fails.
            500: If the tool raises an unexpected exception.
        """
        registry = _ensure_registry()
        if name not in registry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool {name!r} not found. Check GET /utcp for available tools.",
            )
        kwargs: Dict[str, Any] = body or {}
        try:
            result = invoke_tool(name, kwargs, registry=registry)
            return JSONResponse(content=result)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Tool input validation failed: {exc}",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Tool execution failed: {exc}",
            ) from exc

    return router


def __getattr__(name: str) -> Any:
    if name == "utcp_router":
        warnings.warn(
            "ovos_persona_server.utcp.utcp_router is deprecated, use make_utcp_router(); "
            f"it will be removed in {VERSION_MAJOR}.{VERSION_MINOR + 1}.0",
            DeprecationWarning, stacklevel=2)
        return make_utcp_router()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
