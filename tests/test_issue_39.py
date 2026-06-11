"""Tests for issue #39 fixes: declared deps, py3.9 f-string compat."""
import ast
import os
import sys
import textwrap
import tokenize
import io

PYPROJECT = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
CHAT_PY = os.path.join(os.path.dirname(__file__), "..",
                       "ovos_persona_server", "chat.py")


def _load_pyproject_deps():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # py<3.11 fallback
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["dependencies"]


def test_uvicorn_declared():
    deps = _load_pyproject_deps()
    assert any(d.startswith("uvicorn") for d in deps), (
        "uvicorn must be in [project].dependencies"
    )


def test_ovos_workshop_declared():
    deps = _load_pyproject_deps()
    assert any(d.startswith("ovos-workshop") for d in deps), (
        "ovos-workshop must be in [project].dependencies"
    )


def test_no_multiline_fstring_in_chat_py():
    """chat.py must not contain multi-line f-string expressions (requires py>=3.12).

    Multi-line dict/set literals inside an f-string `{}` expression were only
    allowed from Python 3.12 (PEP 701).  We detect this by looking for f-string
    tokens whose content spans more than one line.
    """
    with open(CHAT_PY, encoding="utf-8") as fh:
        source = fh.read()

    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    for tok_type, tok_string, tok_start, tok_end, _ in tokens:
        if tok_type == tokenize.STRING and tok_string.startswith(("f'", 'f"', "f'''", 'f"""')):
            start_line, _ = tok_start
            end_line, _ = tok_end
            if end_line > start_line:
                # Check whether the multi-line span contains a nested { ... }
                # with a newline inside — that's the py3.12-only construct.
                inner = tok_string[2:-1] if tok_string.startswith(('f"', "f'")) else tok_string[4:-3]
                # Simple heuristic: if the f-string body contains a newline between
                # matching braces, it's a multi-line expression inside the f-string.
                depth = 0
                saw_open = False
                for i, ch in enumerate(inner):
                    if ch == '{':
                        depth += 1
                        saw_open = True
                    elif ch == '}':
                        depth -= 1
                    elif ch == '\n' and depth > 0 and saw_open:
                        raise AssertionError(
                            f"chat.py contains a multi-line f-string expression "
                            f"(requires Python 3.12+) starting at line {start_line}.\n"
                            f"Token: {tok_string[:120]!r}..."
                        )


def test_chat_py_parses_on_py39():
    """chat.py source must be parseable on Python 3.9 (no 3.12-only syntax)."""
    with open(CHAT_PY, encoding="utf-8") as fh:
        source = fh.read()
    # ast.parse on any Python version will catch syntax that is invalid on the
    # running interpreter.  Since we run CI on 3.9+, this is a useful smoke test.
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise AssertionError(f"chat.py has a syntax error: {exc}") from exc
