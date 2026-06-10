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

from ovos_persona_server._pkg_version import __version__

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ovos_persona import Persona

import ovos_persona_server.persona


def create_persona_app(persona_path: str) -> FastAPI:
    """Create and configure the FastAPI application for the Persona Server.

    Args:
        persona_path: Path to a persona JSON file.  The ``name`` key is used
            as the persona name; if absent the file's basename is used instead.

    Returns:
        The configured FastAPI application instance.
    """

    with open(persona_path) as f:
        persona = json.load(f)
        persona["name"] = persona.get("name") or os.path.basename(persona_path)

    # TODO - move to dependency injection
    ovos_persona_server.persona.default_persona = persona = Persona(persona["name"], persona)

    app = FastAPI(title="OVOS Persona Server",
                  description="OpenAI/Ollama compatible API for OVOS Personas and Solvers",
                  version=__version__)

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
    from ovos_persona_server.utcp import utcp_router

    app.include_router(chat_router)
    app.include_router(ollama_router)
    app.include_router(utcp_router)

    # Mount MCP server (streamable-HTTP transport) when the `mcp` package is available.
    # mount_mcp_on_app sets streamable_http_path="/" so the endpoint lands at /mcp
    # (not /mcp/mcp) and chains the session manager into the host app lifespan.
    try:
        from ovos_persona_server.mcp_server import mount_mcp_on_app
        mount_mcp_on_app(app)
    except ImportError:
        pass  # mcp extra not installed — UTCP only

    return app
