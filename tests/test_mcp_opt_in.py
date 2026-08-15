"""Tests that MCP mounting is explicit opt-in via ``enable_mcp``/``--mcp``.

Installing the ``mcp`` extra must not, by itself, expose the ``/mcp``
endpoint — the flag is the agreed standard across the OVOS server family
(matching ``ovos-tts-server``). See also tests/e2e/test_e2e_mcp_utcp.py for
the real fastmcp transport behaviour.
"""
import json
from typing import Dict, Generator, List

import pytest

import ovos_persona_server
from ovos_persona_server import create_persona_app
from ovos_persona_server import persona as persona_mod


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
