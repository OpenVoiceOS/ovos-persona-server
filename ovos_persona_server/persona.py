"""
Module for managing and serving persona configurations.

This module defines the FastAPI router for persona-related endpoints,
including loading the default persona and providing its status.
"""

import json
from typing import Any, Dict, Iterable, List, Optional

from fastapi import HTTPException, status
from ovos_bus_client.session import Session, SessionManager
from ovos_persona import Persona
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole, ToolCall

from ovos_persona_server.config import settings

# Dependency injection
default_persona: Optional[Persona] = None

# Registry of every persona served by this process, keyed by persona name.
# Insertion ordered: the first entry is the default unless one was named
# explicitly. Empty in tests that inject a persona via dependency_overrides.
personas: Dict[str, Persona] = {}


class PersonaNoAnswerError(RuntimeError):
    """Raised when no handler in the persona's chain produced an answer.

    ``ovos_persona.solvers.SolverService.chat_completion``/``stream_completion``
    return ``None`` (or yield nothing) when every configured handler declined
    or was never dispatched — e.g. a plugin that loaded but was not wired into
    the chain. That is a legitimate, expected outcome of a misconfigured or
    incomplete persona, not a server bug, so it is raised explicitly here
    rather than left to surface later as an ``AttributeError`` on
    ``None.split()``. Routers catch it and translate it into a response that
    names the cause instead of a generic 500 traceback.
    """


class ModelNotFoundError(HTTPException):
    """Raised when a request names a model that no loaded persona answers to.

    Carries an OpenAI-shaped error object; :func:`create_persona_app` installs a
    handler that renders it as ``{"error": {...}}`` like the OpenAI API does.
    """

    def __init__(self, model: str, available: List[str]) -> None:
        names = ", ".join(available)
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "message": f"The model `{model}` does not exist. "
                           f"Available models: {names}",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            },
        )


def register_personas(loaded: List[Persona], default_name: Optional[str] = None) -> Persona:
    """Populate the process-wide persona registry and pick the default.

    Args:
        loaded: Personas in the order they were given on the command line.
        default_name: Name of the persona to serve when a request omits ``model``.
                      Defaults to the first persona in ``loaded``.

    Returns:
        The default persona.

    Raises:
        ValueError: If ``loaded`` is empty, two personas share a name, or
                    ``default_name`` is not among the loaded personas.
    """
    global default_persona
    if not loaded:
        raise ValueError("at least one persona must be loaded")
    registry: Dict[str, Persona] = {}
    for p in loaded:
        if p.name in registry:
            raise ValueError(f"duplicate persona name: {p.name!r} — "
                             f"persona names are the model ids and must be unique")
        registry[p.name] = p
    if default_name is not None and default_name not in registry:
        raise ValueError(f"--default-persona {default_name!r} is not one of the "
                         f"loaded personas: {', '.join(registry)}")
    personas.clear()
    personas.update(registry)
    default_persona = registry[default_name] if default_name else loaded[0]
    return default_persona


def available_personas(fallback: Optional[Persona] = None) -> List[Persona]:
    """Every persona this process serves, for the model-listing endpoints.

    Falls back to the injected persona when the registry is empty (routers
    mounted standalone in tests, or a persona supplied via
    ``dependency_overrides``).
    """
    if personas:
        return list(personas.values())
    return [fallback] if fallback is not None else []


def multi_persona() -> bool:
    """Whether more than one persona is loaded.

    The ``model`` field only becomes authoritative in multi-persona mode; with a
    single persona it stays advisory, so existing single-persona deployments
    that send ``model="gpt-4"`` keep working.
    """
    return len(personas) > 1


def resolve_persona(model: Optional[str], fallback: Persona,
                    strict: bool = True) -> Persona:
    """Map an incoming ``model`` value to a loaded persona.

    Args:
        model: The model name the client asked for, if any.
        fallback: The default persona (injected by ``get_default_persona``).
        strict: When True an unknown name raises :class:`ModelNotFoundError`.
                When False it falls back to the default — used by the vendor
                surfaces where the model id also selects the response format
                (AWS Bedrock), so a real vendor id must not be an error.

    Returns:
        The persona that must answer this request.
    """
    if not multi_persona() or not model:
        return fallback
    persona = personas.get(model)
    if persona is not None:
        return persona
    if strict:
        raise ModelNotFoundError(model, list(personas))
    return fallback


def memory_session_id(persona: Persona, session_id: str) -> str:
    """Namespace a conversation key by persona when several are served.

    Each persona owns its own memory plugin instance, but a plugin backed by
    shared external storage (a vector DB, a database) keys only by session id,
    so two personas would otherwise read each other's history for the same
    caller. Single-persona deployments keep the bare session id so existing
    stored conversations stay addressable.
    """
    if not multi_persona():
        return session_id
    return f"{persona.name}::{session_id}"


def memory_enabled() -> bool:
    """Whether the chat path transparently applies the persona's memory plugin.

    Server-side deployment toggle (``CHAT_MEMORY``; see :class:`config.Settings`):
    ``"off"`` (default) is a stateless backend, ``"transparent"`` is a single-user
    hosted agent. Read live so it can be flipped/monkeypatched without re-import.
    """
    return (settings.chat_memory or "off").strip().lower() == "transparent"


def _flatten_text(content: Any) -> str:
    """Coerce OpenAI message content (str | content-parts | None) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [(p.get("text") or "") if isinstance(p, dict) else str(p) for p in content]
        return " ".join(p for p in parts if p)
    return str(content)


def _role(raw: Any) -> MessageRole:
    """Map an OpenAI role string to MessageRole (legacy 'function' -> tool)."""
    raw = getattr(raw, "value", raw)
    if raw == "function":
        raw = "tool"
    try:
        return MessageRole(raw)
    except ValueError:
        return MessageRole.USER


def _messages_to_agent(messages: List[Dict[str, Any]]) -> List[AgentMessage]:
    """Convert OpenAI message dicts (incl. assistant tool_calls / tool results) to AgentMessages."""
    out: List[AgentMessage] = []
    for m in messages:
        tool_calls = None
        if m.get("tool_calls"):
            tool_calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(ToolCall(id=tc.get("id") or "", name=fn.get("name", ""), arguments=args))
        out.append(AgentMessage(
            role=_role(m.get("role", "user")),
            content=_flatten_text(m.get("content")),
            tool_calls=tool_calls,
            tool_call_id=m.get("tool_call_id"),
            name=m.get("name") or None,
        ))
    return out


def _last_user_utterance(messages: List[dict]) -> str:
    """Extract the latest user turn's text from OpenAI-style message dicts."""
    for m in reversed(messages):
        if (m.get("role") or "") == "user":
            return _flatten_text(m.get("content"))
    return ""


def run_chat(persona: Persona, messages: List[dict], sess: Optional[Session] = None,
             memory: Optional[bool] = None, session_id: Optional[str] = None) -> str:
    """Call ``persona.chat`` with a Session.

    ovos-persona's ``Persona.chat(messages, sess)`` requires a Session (it reads
    ``sess.lang`` / ``sess.system_unit``). Vendor routers carry no session of
    their own, so a default one is supplied. Centralised here so the ovos-persona
    call contract lives in a single place.

    When ``memory`` is enabled (``CHAT_MEMORY=transparent`` by default) and the
    persona has a memory plugin, the server owns conversation state: only the
    latest user message is treated as the new turn, history/RAG is folded in via
    the persona's ``memory_module`` keyed by ``session_id``, and the exchange is
    persisted for the next call — mirroring the spoken ``query`` path. Otherwise
    the client-supplied messages are passed through unchanged (stateless backend).
    """
    sess = sess or SessionManager().get()
    if memory is None:
        memory = memory_enabled()
    if memory and persona.memory is not None:
        sid = memory_session_id(persona, session_id or sess.session_id)
        utterance = _last_user_utterance(messages)
        context = persona.memory.build_conversation_context(utterance, sid)
        reply = persona.chat(context, sess=sess)
        if reply is None:
            raise PersonaNoAnswerError(
                f"persona {persona.name!r} produced no answer: no handler in "
                f"its chain returned a response for this request")
        persona.memory.update_history(
            [AgentMessage(MessageRole.USER, utterance),
             AgentMessage(MessageRole.ASSISTANT, reply or "")], sid)
        return reply
    reply = persona.chat(_messages_to_agent(messages), sess=sess)
    if reply is None:
        raise PersonaNoAnswerError(
            f"persona {persona.name!r} produced no answer: no handler in "
            f"its chain returned a response for this request")
    return reply


def run_stream(persona: Persona, messages: List[dict], sess: Optional[Session] = None,
               memory: Optional[bool] = None, session_id: Optional[str] = None) -> Iterable[str]:
    """Call ``persona.stream`` with a Session (see :func:`run_chat`).

    Honors the same ``CHAT_MEMORY`` toggle as :func:`run_chat`; in transparent
    mode the streamed tokens are accumulated and persisted to memory once the
    stream completes.
    """
    sess = sess or SessionManager().get()
    if memory is None:
        memory = memory_enabled()
    if memory and persona.memory is not None:
        sid = memory_session_id(persona, session_id or sess.session_id)
        utterance = _last_user_utterance(messages)
        context = persona.memory.build_conversation_context(utterance, sid)

        def _streamer() -> Iterable[str]:
            chunks: List[str] = []
            for tok in persona.stream(context, sess=sess):
                chunks.append(tok)
                yield tok
            if not any(chunks):
                raise PersonaNoAnswerError(
                    f"persona {persona.name!r} produced no answer: no handler "
                    f"in its chain streamed a response for this request")
            persona.memory.update_history(
                [AgentMessage(MessageRole.USER, utterance),
                 AgentMessage(MessageRole.ASSISTANT, "".join(chunks))], sid)

        return _streamer()
    return _require_stream_answer(persona, persona.stream(_messages_to_agent(messages), sess=sess))


def _require_stream_answer(persona: Persona, chunks: Iterable[str]) -> Iterable[str]:
    """Wrap a token stream, raising :class:`PersonaNoAnswerError` if it yields nothing.

    Mirrors the ``None`` check in :func:`run_chat` for the streaming seam:
    ``SolverService.stream_completion`` simply yields no tokens when no handler
    answered, so the "no answer" case has to be detected by watching for an
    empty stream rather than a sentinel return value.
    """
    got_answer = False
    for tok in chunks:
        if tok:
            got_answer = True
        yield tok
    if not got_answer:
        raise PersonaNoAnswerError(
            f"persona {persona.name!r} produced no answer: no handler in its "
            f"chain streamed a response for this request")


async def get_default_persona() -> Persona:
    """
    Asynchronously loads and returns the default persona.

    This function acts as a dependency for FastAPI endpoints, ensuring
    the persona is loaded once and reused across requests.
    It raises HTTPException if the persona file is not configured,
    not found, or invalid.

    Returns:
        Persona: The loaded OVOS Persona instance.

    Raises:
        HTTPException: If persona configuration or loading fails.
    """
    global default_persona
    if default_persona is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to load persona")
    return default_persona
