"""
Module for managing and serving persona configurations.

This module defines the FastAPI router for persona-related endpoints,
including loading the default persona and providing its status.
"""

import json
from typing import Optional

from fastapi import HTTPException, status
from ovos_persona import Persona

from ovos_persona_server.config import settings

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
        persona_data: dict = settings.persona_config
        try:
            default_persona = Persona(persona_data["name"], persona_data)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail=f"JSONDecodeError for persona config: {persona_data}") from e
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail=f"Invalid JSON config for persona: {persona_data}") from e
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Failed to load persona: {e}") from e
    return default_persona
