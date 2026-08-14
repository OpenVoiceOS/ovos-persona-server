"""Unit tests for the `tool_choice` wire-format compatibility of
ChatCompletionRequest: OpenAI's real API accepts `Literal["none", "auto",
"required"]` (see openai.types.chat.chat_completion_tool_choice_option_param),
not the non-standard `"tool"` spelling this server previously required.
"""
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

_TOOLS = [{"type": "function", "function": {
    "name": "calc", "description": "add", "parameters": {"type": "object", "properties": {}}}}]


class _ToolEngine:
    supports_tools = True

    def __init__(self, resp):
        self._resp = resp

    def continue_chat(self, messages, session_id="default", lang=None, units=None, tools=None):
        return self._resp


def _make_app():
    from ovos_persona_server.chat import chat_router
    from ovos_persona_server.persona import get_default_persona

    engine = _ToolEngine(AgentMessage(role=MessageRole.ASSISTANT, content="42"))
    persona = MagicMock()
    persona.name = "tool-choice-persona"
    persona.solvers.modules = [engine]

    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_default_persona] = lambda: persona
    return app


def _post(tool_choice):
    app = _make_app()
    client = TestClient(app)
    return client.post(
        "/openai/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "6*7?"}],
              "tools": _TOOLS, "tool_choice": tool_choice})


def _is_schema_validation_error(response):
    """True if the request was rejected by pydantic enum validation before
    reaching application logic, rather than by a legitimate business-rule
    422 raised once inside the handler (e.g. tool_choice forcing is
    unsupported).
    """
    detail = response.json().get("detail")
    return isinstance(detail, list) and any(
        item.get("loc", [None])[-1] == "str-enum[ChatCompletionToolChoiceOption]"
        for item in detail if isinstance(item, dict))


def test_tool_choice_required_is_accepted():
    """The standard OpenAI spelling must clear schema validation. It is then
    legitimately rejected at the application layer because forcing a tool
    call is unsupported (see tests/test_tool_choice.py) -- that is a
    different, later 422 than the enum-validation one this used to raise.
    """
    r = _post("required")
    assert not _is_schema_validation_error(r), r.json()


def test_tool_choice_none_still_works():
    r = _post("none")
    assert r.status_code != 422, r.json()


def test_tool_choice_auto_still_works():
    r = _post("auto")
    assert r.status_code != 422, r.json()


def test_tool_choice_deprecated_tool_alias_still_works():
    """`tool` is this server's original non-standard spelling; kept as an
    alias so any existing caller relying on it does not break. It clears
    schema validation just like `required` does, and is rejected by the
    same application-layer forcing check.
    """
    r = _post("tool")
    assert not _is_schema_validation_error(r), r.json()
