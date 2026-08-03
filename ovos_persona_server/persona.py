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
        sid = session_id or sess.session_id
        utterance = _last_user_utterance(messages)
        context = persona.memory.build_conversation_context(utterance, sid)
        reply = persona.chat(context, sess=sess)
        persona.memory.update_history(
            [AgentMessage(MessageRole.USER, utterance),
             AgentMessage(MessageRole.ASSISTANT, reply or "")], sid)
        return reply
    return persona.chat(_messages_to_agent(messages), sess=sess)


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
        sid = session_id or sess.session_id
        utterance = _last_user_utterance(messages)
        context = persona.memory.build_conversation_context(utterance, sid)

        def _streamer() -> Iterable[str]:
            chunks: List[str] = []
            for tok in persona.stream(context, sess=sess):
                chunks.append(tok)
                yield tok
            persona.memory.update_history(
                [AgentMessage(MessageRole.USER, utterance),
                 AgentMessage(MessageRole.ASSISTANT, "".join(chunks))], sid)

        return _streamer()
    return persona.stream(_messages_to_agent(messages), sess=sess)


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
