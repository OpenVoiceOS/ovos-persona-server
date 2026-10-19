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
UTCP (Universal Tool Calling Protocol) router.

Provides two endpoints:

``GET /tools/manual``
    Returns a UTCP manual JSON describing all installed OPM tool plugins as
    HTTP-callable tools.  Clients parse this once and know how to call each
    tool.

``POST /tools/{name}``
    Generic invocation endpoint.  The request body is a JSON object whose
    fields are the tool's keyword arguments.  Returns the tool's output as
    JSON.

UTCP manual format reference:
    https://www.utcp.io/docs/for-tool-providers
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from ovos_persona_server.tools import get_flat_tool_registry, invoke_tool, list_tool_schemas

UTCP_VERSION = "1.0.0"
MANUAL_VERSION = "1.0.0"

utcp_router = APIRouter(prefix="/tools", tags=["utcp"])

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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



@utcp_router.get(
    "/manual",
    summary="UTCP manual — list all available tools",
    response_class=JSONResponse,
)
async def utcp_manual(request: Request,
                      request_base_url: Optional[str] = None) -> JSONResponse:
    """
    Return a UTCP 1.0 manual describing all installed OPM tool plugins.

    The tool URLs use the base URL of the incoming request. The
    ``request_base_url`` query parameter replaces it, for a proxy that does
    not forward the public host.

    Example::

        GET /tools/manual?request_base_url=https://myserver.example.com
    """
    base_url = (request_base_url or str(request.base_url)).rstrip("/")
    registry = _ensure_registry()
    schemas = list_tool_schemas(registry)
    return JSONResponse(content={
        "utcp_version": UTCP_VERSION,
        "manual_version": MANUAL_VERSION,
        "tools": [_utcp_tool_entry(base_url, td) for td in schemas],
    })


@utcp_router.post(
    "/{name}",
    summary="Invoke an OPM tool plugin by name",
    response_class=JSONResponse,
)
async def invoke_tool_endpoint(name: str, body: Dict[str, Any] = None) -> JSONResponse:
    """
    Invoke a named OPM tool plugin and return its output as JSON.

    The request body must be a JSON object whose keys match the tool's
    declared input parameters (see ``GET /tools/manual`` for schemas).

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
            detail=f"Tool {name!r} not found. Check GET /tools/manual for available tools.",
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
