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

from ovos_persona_server.chat import chat_router
from ovos_persona_server.config import settings
from ovos_persona_server.embeddings import embeddings_router, get_text_embeddings, get_image_embeddings
from ovos_persona_server.files import files_router
from ovos_persona_server.metadata import init_db
from ovos_persona_server.ollama import ollama_router
from ovos_persona_server.persona import persona_router, get_default_persona
from ovos_persona_server.vector_stores import vector_stores_router
from ovos_persona_server.version import VERSION_MAJOR, VERSION_ALPHA, VERSION_BUILD, VERSION_MINOR


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Initialize the unified SQLite database before the application starts; no shutdown actions are performed.
    
    This async lifespan context manager calls init_db() prior to yielding control to the FastAPI application so startup-dependent resources are ready. It yields once to allow the app to run and performs no explicit cleanup on shutdown.
    
    Returns:
        None: yielded once to signal startup completion.
    """
    await init_db() # Call init_db from database.py
    yield
    # No specific shutdown logic needed


def create_persona_app(persona_path: Optional[str] = None) -> FastAPI:
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
    app = FastAPI(title="OVOS Persona Server",
                  description="OpenAI/Ollama compatible API for OVOS Personas and Solvers",
                  version=version_str,
                  lifespan=lifespan)

    # TODO - from .env
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
        allow_headers=["*"],  # Allows all headers
    )

    # Include routers for different API functionalities
    app.include_router(persona_router)
    app.include_router(chat_router)
    app.include_router(embeddings_router)
    app.include_router(ollama_router)
    app.include_router(files_router)
    app.include_router(vector_stores_router)

    return app