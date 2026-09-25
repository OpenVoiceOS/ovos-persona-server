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
Tool-plugin discovery and invocation helpers.

Discovers all installed OPM ``ToolBox`` plugins (entry-point group
``opm.agents.toolbox``) and provides a flat, name-keyed registry that the
MCP and UTCP layers can query without depending on each other.
"""

from typing import Any, Dict, List, Optional, Tuple

from ovos_utils.log import LOG

from ovos_persona_server.config import settings

# PluginTypes.AGENT_TOOLBOX → "opm.agents.toolbox"
try:
    from ovos_plugin_manager.utils import find_plugins, PluginTypes

    _TOOLBOX_ENTRY_POINT = PluginTypes.AGENT_TOOLBOX  # type: ignore[attr-defined]
except Exception:  # pragma: no cover – graceful fallback when OPM not installed
    find_plugins = None  # type: ignore[assignment]
    _TOOLBOX_ENTRY_POINT = "opm.agents.toolbox"


def _load_toolboxes() -> List[Any]:
    """
    Instantiate every installed ``ToolBox`` plugin.

    Per the OPM ``ToolBox`` contract, ``toolbox_id`` is supplied by each
    plugin's own ``__init__`` (via ``super().__init__(toolbox_id=...)``), not
    passed in by the loader, so the loader makes exactly one call:
    ``cls(config=cfg, bus=bus)``. There is no per-toolbox
    section in :class:`~ovos_persona_server.config.Settings` today, so the
    plugin is handed the same top-level persona config blob solver plugins
    already receive (``settings.persona_config``, keyed by plugin name via
    the ``self.llm_solver: {...}`` convention) — a well-behaved toolbox can
    look up its own ``toolbox_id`` key there for plugin-specific options,
    the same way solvers look up theirs. This server has no message bus of
    its own (it is a plain FastAPI process), so ``bus`` is always ``None``.

    Returns:
        List of live ``ToolBox`` instances.  Plugins that fail to instantiate
        are skipped with a warning so one bad plugin cannot block the others.
    """
    if find_plugins is None:
        LOG.warning("ovos-plugin-manager not available; no tool plugins loaded")
        return []

    plugin_classes: Dict[str, Any] = find_plugins(_TOOLBOX_ENTRY_POINT)
    instances: List[Any] = []
    cfg = settings.persona_config
    bus = None
    for name, cls in plugin_classes.items():
        try:
            # Hand the plugin its OWN config section when the persona defines
            # one, keyed by plugin name -- the same shape solver plugins get.
            # Both existing ToolBox implementations (ovos-mcp-toolbox and
            # ovos-utcp-toolbox) read their settings straight off `config`
            # (`config["transport"]`, `config["command"]`, `config["url"]`),
            # so handing them the whole persona blob meant they could never be
            # configured at all: MCPToolBox raised KeyError('command'), which
            # discover_tools() swallows into a warning, and the persona simply
            # reported zero tools. The full blob is still passed when no
            # section exists, so a toolbox that self-locates keeps working.
            plugin_cfg = cfg.get(name) if isinstance(cfg, dict) else None
            instance = cls(config=plugin_cfg if isinstance(plugin_cfg, dict) else cfg,
                           bus=bus)
            instances.append(instance)
            LOG.debug("Loaded ToolBox plugin: %s", name)
        except Exception as exc:
            LOG.warning("Failed to load ToolBox plugin %s: %s", name, exc)
    return instances


def get_flat_tool_registry() -> Dict[str, Tuple[Any, Any]]:
    """
    Build a flat ``{tool_name: (toolbox, agent_tool)}`` registry from all
    installed ToolBox plugins.

    When two plugins export a tool with the same name the entry is prefixed
    with ``<toolbox_id>__<tool_name>`` to avoid collisions.

    Returns:
        Dict mapping unique tool names to ``(toolbox_instance, AgentTool)`` tuples.
    """
    registry: Dict[str, Tuple[Any, Any]] = {}
    for toolbox in _load_toolboxes():
        try:
            toolbox.refresh_tools()
        except Exception:
            pass
        for tool_name, tool in toolbox.tools.items():
            if tool_name in registry:
                # Collision — use qualified name for both
                existing_tb, existing_tool = registry.pop(tool_name)
                qualified_existing = f"{existing_tb.toolbox_id}__{tool_name}"
                registry[qualified_existing] = (existing_tb, existing_tool)
                qualified_new = f"{toolbox.toolbox_id}__{tool_name}"
                registry[qualified_new] = (toolbox, tool)
            else:
                registry[tool_name] = (toolbox, tool)
    return registry


def invoke_tool(tool_name: str, kwargs: Dict[str, Any],
                registry: Optional[Dict[str, Tuple[Any, Any]]] = None) -> Dict[str, Any]:
    """
    Invoke a named tool and return its output as a plain dict.

    Args:
        tool_name: Registered tool name (possibly prefixed with toolbox id).
        kwargs: Raw keyword arguments forwarded to the tool.
        registry: Pre-built registry; if *None* a fresh one is constructed.

    Returns:
        JSON-serialisable dict produced by ``ToolOutput.model_dump``.

    Raises:
        KeyError: If *tool_name* is not found in the registry.
        Exception: Propagated from the underlying tool execution.
    """
    if registry is None:
        registry = get_flat_tool_registry()
    if tool_name not in registry:
        raise KeyError(f"Unknown tool: {tool_name!r}")
    toolbox, _ = registry[tool_name]
    result = toolbox.call_tool(tool_name.split("__", 1)[-1], kwargs)
    return result.model_dump(mode="json")


def list_tool_schemas(registry: Optional[Dict[str, Tuple[Any, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Return a list of tool descriptor dicts suitable for embedding in API
    responses or the MCP tool manifest.

    Each dict has the keys ``name``, ``description``, ``argument_schema``
    (JSON Schema dict) and ``output_schema`` (JSON Schema dict).

    Args:
        registry: Pre-built registry; if *None* a fresh one is constructed.

    Returns:
        List of tool descriptor dicts.
    """
    if registry is None:
        registry = get_flat_tool_registry()
    schemas: List[Dict[str, Any]] = []
    for registered_name, (_, tool) in registry.items():
        schemas.append({
            "name": registered_name,
            "description": tool.description,
            "argument_schema": tool.argument_schema.model_json_schema(),
            "output_schema": tool.output_schema.model_json_schema(),
        })
    return schemas
