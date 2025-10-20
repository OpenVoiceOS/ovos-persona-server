"""
Main application entry point for the OVOS Persona Server.

This module initializes the FastAPI application, sets up CORS middleware,
and includes various API routers for chat, embeddings, Ollama, persona status,
and mock OpenAI Vector Stores. It now centrally manages the unified SQLite database
initialization using SQLAlchemy.
"""
import os
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ovos_utils.log import LOG

from ovos_persona_server.a2a_executor import get_a2a_app
from ovos_persona_server.config import settings
from ovos_persona_server.persona import get_default_persona
from ovos_persona_server.version import VERSION_MAJOR, VERSION_ALPHA, VERSION_BUILD, VERSION_MINOR


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages the lifespan of the FastAPI application, ensuring the default persona is loaded.
    """
    await get_default_persona()
    yield
    # No specific shutdown logic needed


def create_persona_app(persona_path: str, domain: Optional[str] = None, enable_a2a: bool = False) -> FastAPI:
    """
    Creates and configures the FastAPI application for the Persona Server.

    Args:
        persona_path (Optional[str]): Optional path to a persona JSON file.
                                      If provided, it overrides the default
                                      persona path from settings or environment.

    Returns:
        FastAPI: The configured FastAPI application instance.
    """
    if persona_path:
        settings.persona = os.path.expanduser(persona_path)

    version_str = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}"
    if VERSION_ALPHA:
        version_str += f"a{VERSION_ALPHA}"

    title = "OpenVoiceOS Persona Server"
    description = "OpenAI/Ollama compatible API for OVOS Personas and Solvers"

    if enable_a2a:
        app = get_a2a_app(version=version_str,
                          domain=domain,
                          title=title,
                          description=description,
                          lifespan=lifespan)
        LOG.info("Enabled A2A endpoints")
    else:
        app = FastAPI(title=title,
                      description=description,
                      version=version_str,
                      lifespan=lifespan)

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
