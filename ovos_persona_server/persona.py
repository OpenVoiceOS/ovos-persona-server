"""
Module for managing and serving persona configurations.

This module defines the FastAPI router for persona-related endpoints,
including loading the default persona and providing its status.
"""

from typing import Optional

from fastapi import HTTPException, status
from ovos_persona import Persona

# Dependency injection
default_persona: Optional[Persona] = None


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
