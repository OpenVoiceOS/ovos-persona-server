"""
Main application entry point for the OVOS Persona Server.

This module initializes the FastAPI application, sets up CORS middleware,
and includes various API routers for chat, embeddings, Ollama, persona status,
and mock OpenAI Vector Stores. It now centrally manages the unified SQLite database
initialization using SQLAlchemy.
"""
import json
import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ovos_persona import Persona

import ovos_persona_server.persona


def create_persona_app(persona_path: str, a2a_base_url: Optional[str] = None) -> FastAPI:
    """
    Creates and configures the FastAPI application for the Persona Server.

    Args:
        persona_path: Path to a persona JSON file.
        a2a_base_url: If provided, mounts an A2A-compatible endpoint at ``/a2a``
                      using this URL as the public base URL in the Agent Card
                      (e.g. ``http://myhost:8337/a2a``). Requires ``a2a-sdk``
                      to be installed (``uv pip install 'ovos-persona-server[a2a]'``).

    Returns:
        FastAPI: The configured FastAPI application instance.
    """

    with open(persona_path) as f:
        persona = json.load(f)
        persona["name"] = persona.get("name") or os.path.basename(persona_path)

    # TODO - move to dependency injection
    ovos_persona_server.persona.default_persona = persona = Persona(persona["name"], persona)

    from ovos_persona_server.version import VERSION_MAJOR, VERSION_ALPHA, VERSION_BUILD, VERSION_MINOR

    version_str = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}"
    if VERSION_ALPHA:
        version_str += f"a{VERSION_ALPHA}"

    app = FastAPI(title="OVOS Persona Server",
                  description="OpenAI/Ollama compatible API for OVOS Personas and Solvers",
                  version=version_str)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
        allow_headers=["*"],  # Allows all headers
    )

    # Include routers for different API functionalities
    # imported here only after the Persona object is loaded
    from ovos_persona_server.chat import chat_router
    from ovos_persona_server.ollama import ollama_router
    from ovos_persona_server.deprecated_routers import (
        add_deprecation_middleware,
        register_deprecated_routes,
    )

    # Canonical prefixed routers
    app.include_router(chat_router)         # /openai/v1/...
    app.include_router(ollama_router)       # /ollama/api/...

    # Legacy deprecated paths — same handlers, with Deprecation + Link headers
    register_deprecated_routes(app)         # /v1/... and /api/... (deprecated)
    add_deprecation_middleware(app)         # injects headers on /v1/* and /api/*

    # Optional A2A endpoint — mounted only when a2a_base_url is provided
    if a2a_base_url is not None:
        from ovos_persona_server.a2a import _A2A_AVAILABLE, create_a2a_application
        if _A2A_AVAILABLE:
            a2a_starlette = create_a2a_application(persona, a2a_base_url).build()
            app.mount("/a2a", a2a_starlette)
        else:
            import logging
            logging.getLogger(__name__).warning(
                "a2a_base_url was set but a2a-sdk is not installed — "
                "A2A endpoint will not be available. "
                "Install with: uv pip install 'ovos-persona-server[a2a]'"
            )

    return app
