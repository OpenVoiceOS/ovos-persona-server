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
Server-side (persona-owned) tool participation for chat completions.

Function calling in OVOS is modelled through the ``ChatEngine`` abstraction:
a tool-capable engine (``supports_tools=True``) accepts a ``tools`` argument on
``continue_chat`` and returns an ``AgentMessage`` whose ``tool_calls`` lists the
calls the model requested. Tools reach the engine as OpenAI ``tools`` specs (the
neutral interchange format), whether they came from the API client or from a
persona's own ``ToolBox`` plugins.

Two kinds of tools coexist in one request:

* **Client-side** tools — supplied in the OpenAI ``tools`` field. The *caller*
  executes them; the server relays the model's ``tool_calls`` back with
  ``finish_reason="tool_calls"`` and the client replies with ``role:"tool"``
  results on the next turn.
* **Server-side** tools — the persona's installed ``ToolBox`` plugins. The
  *server* executes them: when the model calls one, this module runs it via
  :func:`ovos_persona_server.tools.invoke_tool`, appends the result as a
  ``role:"tool"`` message, and re-invokes the engine — an agentic loop — until
  the model answers or requests a client-side tool.

This keeps everything on the OVOS abstraction (``continue_chat`` /
``AgentMessage`` / ``ToolBox``); nothing here talks to a provider endpoint
directly.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole, ToolCall
from ovos_utils.log import LOG

from ovos_persona_server.tools import get_flat_tool_registry, invoke_tool

# Upper bound on server-side tool-execution rounds, so a model that keeps
# calling a server tool can never spin forever.
MAX_TOOL_ITERS: int = 5

# ToolBox plugins are effectively frozen until the process restarts (mirroring
# the MCP/UTCP layers), so the flat registry is built once and cached.
_REGISTRY_CACHE: Optional[Dict[str, Tuple[Any, Any]]] = None


def cached_registry() -> Dict[str, Tuple[Any, Any]]:
    """Return the flat ToolBox registry, building it once and caching it.

    Avoids re-instantiating every installed ToolBox plugin on each chat request.
    Call :func:`reset_registry_cache` to force a rebuild (used by tests).
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        try:
            _REGISTRY_CACHE = get_flat_tool_registry()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Failed to build server tool registry: %s", exc)
            _REGISTRY_CACHE = {}
    return _REGISTRY_CACHE


def reset_registry_cache() -> None:
    """Drop the cached ToolBox registry so the next call rebuilds it."""
    global _REGISTRY_CACHE
    _REGISTRY_CACHE = None


def build_server_tool_specs(
        registry: Optional[Dict[str, Tuple[Any, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], set]:
    """Return the persona's own ToolBox tools as OpenAI specs plus their names.

    Args:
        registry: Pre-built flat tool registry; if *None* one is constructed
            from every installed ToolBox plugin.

    Returns:
        ``(specs, names)`` where *specs* is a list of OpenAI
        ``{"type": "function", "function": {...}}`` dicts and *names* is the set
        of tool names the server owns (and will execute itself).
    """
    if registry is None:
        try:
            registry = get_flat_tool_registry()
        except Exception as exc:  # noqa: BLE001 - one bad plugin must not break chat
            LOG.warning("Failed to build server tool registry: %s", exc)
            registry = {}
    specs: List[Dict[str, Any]] = []
    names: set = set()
    for registered_name, (_, tool) in registry.items():
        try:
            parameters = tool.argument_schema.model_json_schema()
        except Exception:  # noqa: BLE001
            parameters = {"type": "object", "properties": {}}
        specs.append({
            "type": "function",
            "function": {
                "name": registered_name,
                "description": getattr(tool, "description", "") or "",
                "parameters": parameters,
            },
        })
        names.add(registered_name)
    return specs, names


def _execute_server_call(tc: ToolCall, registry: Dict[str, Tuple[Any, Any]]) -> AgentMessage:
    """Run one server-side tool call and wrap its result as a TOOL message."""
    try:
        result = invoke_tool(tc.name, tc.arguments, registry=registry)
        content = json.dumps(result)
    except Exception as exc:  # noqa: BLE001 - surface the failure to the model
        LOG.error("Server tool %r failed: %s", tc.name, exc)
        content = json.dumps({"error": str(exc)})
    return AgentMessage(role=MessageRole.TOOL, content=content,
                        tool_call_id=tc.id, name=tc.name)


def run_tool_loop(
        engine: Any,
        messages: List[AgentMessage],
        client_specs: List[Dict[str, Any]],
        session_id: str = "default",
        lang: Optional[str] = None,
        units: Optional[str] = None,
        registry: Optional[Dict[str, Tuple[Any, Any]]] = None,
) -> AgentMessage:
    """Drive a tool-capable engine, executing the persona's own tools server-side.

    Client tools and the persona's ToolBox tools are offered to the model
    together. Calls to server tools are executed here and fed back to the model;
    the loop ends when the model returns content or requests a client-side tool
    (which is returned unexecuted for the caller to run).

    Args:
        engine: A ``supports_tools=True`` ChatEngine.
        messages: Conversation so far as ``AgentMessage`` objects.
        client_specs: OpenAI tool specs supplied by the API client (executed by
            the client, relayed back verbatim).
        session_id: Session identifier passed to the engine.
        lang: BCP-47 language code.
        units: Preferred unit system.
        registry: Pre-built server tool registry; built on demand when *None*.

    Returns:
        The final assistant ``AgentMessage``. Its ``tool_calls`` are non-empty
        only for client-side tools the caller must execute.
    """
    if registry is None:
        try:
            registry = get_flat_tool_registry()
        except Exception:  # noqa: BLE001
            registry = {}
    server_specs, server_names = build_server_tool_specs(registry)
    # A client tool sharing a name with one of the persona's own tools would
    # otherwise be offered to the model twice, and the call would be executed
    # server-side rather than relayed -- silently running something other than
    # what the caller asked for. The server tool wins (it is the one that will
    # actually execute), and the shadowed client spec is dropped so the model
    # never sees a duplicate name.
    shadowed = [spec for spec in client_specs
                if spec.get("function", {}).get("name") in server_names]
    if shadowed:
        LOG.warning("client tool(s) %s shadowed by persona tools of the same "
                    "name; the persona's tools will be used",
                    [spec["function"]["name"] for spec in shadowed])
    offered = [spec for spec in client_specs
               if spec.get("function", {}).get("name") not in server_names]
    offered += server_specs

    convo = list(messages)
    resp: AgentMessage = AgentMessage(role=MessageRole.ASSISTANT, content="")
    for _ in range(MAX_TOOL_ITERS):
        resp = engine.continue_chat(convo, session_id=session_id, lang=lang,
                                    units=units, tools=offered or None)
        calls = getattr(resp, "tool_calls", None) or []
        server_calls = [c for c in calls if c.name in server_names]
        # No server-side work to do: either plain content, or client tool_calls
        # the caller must execute. Return as-is.
        if not server_calls:
            return resp
        # If the model mixed a client tool into the batch we cannot resolve the
        # turn server-side (the client tool is ours to relay, not run). Relay the
        # whole assistant message so the caller can execute what it owns.
        if len(server_calls) != len(calls):
            return resp
        # Execute every server-side call, append results, and continue the loop.
        convo.append(AgentMessage(role=MessageRole.ASSISTANT,
                                  content=resp.content or "",
                                  tool_calls=calls))
        for tc in server_calls:
            convo.append(_execute_server_call(tc, registry))
    LOG.warning("Server tool loop hit MAX_TOOL_ITERS=%d; returning last response",
                MAX_TOOL_ITERS)
    return resp
