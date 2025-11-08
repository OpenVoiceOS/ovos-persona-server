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


def create_persona_app(persona_path: str) -> FastAPI:
    """
    Creates and configures the FastAPI application for the Persona Server.

    Args:
        persona_path (Optional[str]): Optional path to a persona JSON file.
                                      If provided, it overrides the default
                                      persona path from settings or environment.

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

    app.include_router(chat_router)
    app.include_router(ollama_router)


    return app
