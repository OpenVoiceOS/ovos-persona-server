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
"""Every plugin and package the documentation names must be installable.

The documentation is the only interface some readers use, so a name that
resolves to nothing is a defect they meet before any code runs. Two kinds of
name are checked: the packages on a ``pip install`` line, including the lines
inside dockerfile blocks, and the plugin names inside a persona's ``solvers``
or ``handlers`` list. A plugin name is usually its package name; where it is
not, the exception is listed here with the package that provides it, so a new
unlisted name fails until somebody says which package installs it.

Run in isolation::

    pytest tests/e2e/test_e2e_documented_packages.py -v
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# A pip invocation, its options, and the names that follow, up to the end of the
# command. Line continuations are joined first, so one match carries them all.
_INSTALL = re.compile(r"\b(?:uv\s+)?pip\s+install\b([^\n]*)")
_OPTION = re.compile(r"^-")
# A requirement as documentation writes it: a name, optionally with extras and a
# version specifier, and nothing that makes it a path or a URL.
_REQUIREMENT = re.compile(r"^([A-Za-z][A-Za-z0-9._-]*)(\[[^\]]*\])?([<>=!~].*)?$")
# A persona's plugin list, as the documentation writes it in a json block.
_PLUGIN_LIST = re.compile(r'"(?:solvers|handlers|toolboxes)"\s*:\s*\[(.*?)\]', re.S)
# A plugin selected by a single key rather than a list.
_PLUGIN_KEY = re.compile(r'"(?:memory_module|toolbox)"\s*:\s*"([^"]+)"')
_QUOTED = re.compile(r'"([^"]+)"')
# An MCP client's server list. Its keys are labels the reader chooses, not
# plugin names, so they are removed before the scan below.
_MCP_SERVERS = re.compile(r'"mcpServers"\s*:\s*\{.*?\n\s*\}', re.S)

# Plugin names whose package is named differently. The value is the package
# that installs the plugin, and it is checked against the index like any other.
PLUGIN_PACKAGES = {
    "ovos-chat-openai-plugin": "ovos-openai-plugin",
    "ovos-a2a-solver": "ovos-a2a-solver-plugin",
    "ovos-openai-rag-memory-plugin": "ovos-openai-plugin",
    "ovos-wikipedia-tools": "ovos-wikipedia-plugin",
}


def _markdown_files() -> list[Path]:
    return sorted([REPO / "README.md", *(REPO / "docs").glob("**/*.md")])


def _documented_packages(text: str) -> set[str]:
    names: set[str] = set()
    for tail in _INSTALL.findall(text.replace("\\\n", " ")):
        for token in tail.split():
            token = token.strip("'\"`")
            if not token or _OPTION.match(token):
                continue
            match = _REQUIREMENT.match(token)
            if match:
                names.add(match.group(1).lower())
    return names


def _documented_plugins(text: str) -> set[str]:
    text = _MCP_SERVERS.sub(" ", text)
    names: set[str] = set()
    for block in _PLUGIN_LIST.findall(text):
        names.update(name.lower() for name in _QUOTED.findall(block))
    names.update(name.lower() for name in _PLUGIN_KEY.findall(text))
    # A plugin's own config lives under a key equal to its name, so any quoted
    # ovos name used as a json key is a plugin name the reader must get right.
    names.update(name.lower() for name in re.findall(r'"(ovos-[a-z0-9-]+)"\s*:\s*\{', text))
    return names


def _on_pypi(name: str) -> bool:
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{name}/json", timeout=30
        ) as response:
            return bool(json.loads(response.read())["info"]["name"])
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return False
        raise


def test_the_documentation_names_packages_that_exist():
    found = {path: _documented_packages(path.read_text()) for path in _markdown_files()}
    assert any(found.values()), "no pip install line was found; the parser is broken"

    missing = {
        f"{path.relative_to(REPO)}: {name}"
        for path, names in found.items()
        for name in sorted(names)
        if name != "ovos-persona-server" and not _on_pypi(name)
    }
    assert not missing, "documented but absent from PyPI: " + ", ".join(sorted(missing))


def test_the_documentation_names_plugins_that_can_be_installed():
    found = {path: _documented_plugins(path.read_text()) for path in _markdown_files()}
    assert any(found.values()), "no persona plugin list was found; the parser is broken"

    missing = {
        f"{path.relative_to(REPO)}: {name}"
        for path, names in found.items()
        for name in sorted(names)
        if not _on_pypi(PLUGIN_PACKAGES.get(name, name))
    }
    assert not missing, "no package installs these documented plugins: " + ", ".join(sorted(missing))


def test_every_listed_exception_names_a_real_package():
    absent = sorted(p for p in set(PLUGIN_PACKAGES.values()) if not _on_pypi(p))
    assert not absent, f"PLUGIN_PACKAGES points at packages that do not exist: {absent}"


@pytest.mark.parametrize("name", ["ovos-solver-ddg-plugin", "definitely-not-a-package-x9"])
def test_the_probe_can_see_a_missing_package(name):
    """Without this the test above passes when every lookup silently succeeds."""
    assert _on_pypi(name) is False
