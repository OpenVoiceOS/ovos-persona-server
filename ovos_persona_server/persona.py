"""
Module for managing and serving persona configurations.

This module defines the FastAPI router for persona-related endpoints,
including loading the default persona and providing its status.
"""

from typing import Iterable, List, Optional

from fastapi import HTTPException, status
from ovos_bus_client.session import Session, SessionManager
from ovos_persona import Persona

# Dependency injection
default_persona: Optional[Persona] = None


def run_chat(persona: Persona, messages: List[dict], sess: Optional[Session] = None) -> str:
    """Call ``persona.chat`` with a Session.

    ovos-persona's ``Persona.chat(messages, sess)`` requires a Session (it reads
    ``sess.lang`` / ``sess.system_unit``). Vendor routers carry no session of
    their own, so a default one is supplied. Centralised here so the ovos-persona
    call contract lives in a single place.
    """
    return persona.chat(messages, sess=sess or SessionManager().get())


def run_stream(persona: Persona, messages: List[dict],
               sess: Optional[Session] = None) -> Iterable[str]:
    """Call ``persona.stream`` with a Session (see :func:`run_chat`)."""
    return persona.stream(messages, sess=sess or SessionManager().get())


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
