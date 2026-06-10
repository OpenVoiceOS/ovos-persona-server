# Licensed under the Apache License, Version 2.0
"""Package version string, importable without any heavy runtime dependencies.

This module parses ``version.py`` at import time so that build backends
(``python -m build``) can resolve ``__version__`` without importing the full
package (which would require FastAPI, pydantic, etc. to be installed in the
isolated build environment).

At runtime the same attribute is available on ``ovos_persona_server.__version__``.
"""
import os as _os
import re as _re

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_VERSION_FILE = _os.path.join(_HERE, "version.py")

_BLOCK: dict = {}
with open(_VERSION_FILE) as _f:
    for _line in _f:
        _m = _re.match(r"^(VERSION_MAJOR|VERSION_MINOR|VERSION_BUILD|VERSION_ALPHA)\s*=\s*(\d+)", _line)
        if _m:
            _BLOCK[_m.group(1)] = int(_m.group(2))

__version__: str = (
    f"{_BLOCK['VERSION_MAJOR']}.{_BLOCK['VERSION_MINOR']}.{_BLOCK['VERSION_BUILD']}"
    + (f"a{_BLOCK['VERSION_ALPHA']}" if _BLOCK.get("VERSION_ALPHA") else "")
)
