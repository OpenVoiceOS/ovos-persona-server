"""A ToolBox plugin must receive its own config section, not the whole persona.

Both shipped ToolBox implementations read their settings directly off
``config`` -- ``MCPToolBox`` wants ``transport``/``command``/``url``, and
``UTCPToolBox`` likewise. Handing them the entire persona blob means those keys
are missing, the constructor raises, ``discover_tools`` swallows it into a
warning, and the persona quietly serves zero tools.
"""
import json
from unittest.mock import patch

import pytest

from ovos_persona_server import persona as persona_mod
from ovos_persona_server import tools as tools_mod


@pytest.fixture(autouse=True)
def _no_registered_persona(monkeypatch):
    """Each test chooses its own config source; no persona leaks in."""
    monkeypatch.setattr(persona_mod, "default_persona", None)


class _RecordingToolBox:
    """Stands in for a real ToolBox; records the config it was handed."""

    last_config = None

    def __init__(self, config=None, bus=None):
        type(self).last_config = config
        # Mirrors the real plugins: settings are read straight off `config`.
        self.transport = config["transport"]
        self.tools = {}

    def refresh_tools(self):
        pass


PERSONA = {
    "name": "p",
    "handlers": ["ovos-chat-openai-plugin"],
    "ovos-chat-openai-plugin": {"model": "x"},
    "ovos-mcp-toolbox": {"transport": "stdio", "command": "/bin/true"},
}


def test_toolbox_gets_its_own_section():
    _RecordingToolBox.last_config = None
    with patch.object(tools_mod, "find_plugins",
                      return_value={"ovos-mcp-toolbox": _RecordingToolBox}), \
         patch.object(type(tools_mod.settings), "persona_config", PERSONA):
        boxes = tools_mod._load_toolboxes()

    assert boxes, "toolbox failed to instantiate -- it got the wrong config"
    assert _RecordingToolBox.last_config == PERSONA["ovos-mcp-toolbox"]


def test_toolbox_without_a_section_still_gets_the_blob():
    """A toolbox that self-locates inside the persona config keeps working."""
    class _Blob:
        seen = None

        def __init__(self, config=None, bus=None):
            type(self).seen = config
            self.tools = {}

        def refresh_tools(self):
            pass

    with patch.object(tools_mod, "find_plugins",
                      return_value={"ovos-other-toolbox": _Blob}), \
         patch.object(type(tools_mod.settings), "persona_config", PERSONA):
        tools_mod._load_toolboxes()

    assert _Blob.seen == PERSONA


def test_toolbox_config_comes_from_the_cli_persona(monkeypatch, tmp_path):
    """``--persona file.json`` configures the toolboxes without PERSONA_PATH.

    The server loads its personas from the CLI flags, and the toolbox
    sections live in that same file. Reading them from PERSONA_PATH only
    left every toolbox unconfigured when the user passed just the flag.
    """
    from ovos_persona_server import create_persona_app

    persona_file = tmp_path / "p.json"
    persona_file.write_text(json.dumps(PERSONA))
    monkeypatch.delenv("PERSONA_PATH", raising=False)
    monkeypatch.setattr(tools_mod.settings, "persona", "")
    create_persona_app(str(persona_file))

    _RecordingToolBox.last_config = None
    with patch.object(tools_mod, "find_plugins",
                      return_value={"ovos-mcp-toolbox": _RecordingToolBox}):
        boxes = tools_mod._load_toolboxes()

    assert boxes, "toolbox failed to instantiate -- it got the wrong config"
    assert _RecordingToolBox.last_config == PERSONA["ovos-mcp-toolbox"]
