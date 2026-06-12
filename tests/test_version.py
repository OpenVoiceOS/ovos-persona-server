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
Tests for the dynamic-version contract (pyproject.toml → version.py).

Guards that:
 - ``ovos_persona_server.version.__version__`` is importable and non-empty.
 - ``__version__`` is correctly derived from the VERSION_* integer fields.
 - The version string follows PEP-440 (numeric dotted with optional alpha suffix).
"""

import re

import pytest


class TestVersionModule:
    def test_version_importable(self):
        """version module must be importable without errors."""
        import ovos_persona_server.version as v  # noqa: F401

    def test_version_string_non_empty(self):
        from ovos_persona_server.version import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_version_fields_are_integers(self):
        from ovos_persona_server.version import (
            VERSION_MAJOR, VERSION_MINOR, VERSION_BUILD,
        )
        assert isinstance(VERSION_MAJOR, int)
        assert isinstance(VERSION_MINOR, int)
        assert isinstance(VERSION_BUILD, int)

    def test_version_alpha_is_int_or_zero(self):
        """VERSION_ALPHA must be a non-negative integer (0 means no alpha suffix)."""
        from ovos_persona_server.version import VERSION_ALPHA
        assert isinstance(VERSION_ALPHA, int)
        assert VERSION_ALPHA >= 0

    def test_version_string_matches_version_fields(self):
        """__version__ must be built from VERSION_MAJOR.MINOR.BUILD[aN]."""
        from ovos_persona_server.version import (
            VERSION_MAJOR, VERSION_MINOR, VERSION_BUILD, VERSION_ALPHA, __version__,
        )
        expected_base = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}"
        if VERSION_ALPHA:
            expected = f"{expected_base}a{VERSION_ALPHA}"
        else:
            expected = expected_base
        assert __version__ == expected, (
            f"__version__={__version__!r} does not match computed {expected!r}"
        )

    def test_version_pep440_format(self):
        """__version__ must match PEP-440 format: X.Y.Z or X.Y.ZaN."""
        from ovos_persona_server.version import __version__
        pattern = re.compile(r"^\d+\.\d+\.\d+(a\d+)?$")
        assert pattern.match(__version__), (
            f"__version__={__version__!r} does not match PEP-440 pattern"
        )

    def test_version_major_minor_build_non_negative(self):
        """Version components must be non-negative integers."""
        from ovos_persona_server.version import (
            VERSION_MAJOR, VERSION_MINOR, VERSION_BUILD,
        )
        assert VERSION_MAJOR >= 0
        assert VERSION_MINOR >= 0
        assert VERSION_BUILD >= 0

    def test_stable_version_has_no_alpha_suffix(self):
        """When VERSION_ALPHA is 0, __version__ must not contain 'a'."""
        from ovos_persona_server.version import VERSION_ALPHA, __version__
        if VERSION_ALPHA == 0:
            assert "a" not in __version__

    def test_alpha_version_suffix_matches_alpha_field(self):
        """When VERSION_ALPHA > 0, __version__ must end with 'a{VERSION_ALPHA}'."""
        from ovos_persona_server.version import VERSION_ALPHA, __version__
        if VERSION_ALPHA > 0:
            assert __version__.endswith(f"a{VERSION_ALPHA}")
