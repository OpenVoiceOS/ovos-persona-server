"""
Main application entry point for the OVOS Persona Server.

This module initializes the FastAPI application, sets up CORS middleware,
and includes various API routers for chat, embeddings, Ollama, persona status,
and mock OpenAI Vector Stores. It now centrally manages the unified SQLite database
initialization using SQLAlchemy.
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ovos_persona import Persona

import ovos_persona_server.persona


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise the embeddings/files/vector-store SQLite schema on startup.

    The RAG subsystem is optional; if its dependencies are unavailable the
    server still starts and only the RAG endpoints are unusable.
    """
    try:
        from ovos_persona_server.metadata import init_db
        await init_db()
    except Exception as exc:  # pragma: no cover - optional subsystem
        logging.getLogger(__name__).warning("RAG database init skipped: %s", exc)
    yield


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

    if persona_path:
        with open(persona_path) as f:
            persona = json.load(f)
        persona["name"] = persona.get("name") or os.path.basename(persona_path)
    else:
        # No persona file: fall back to the env-var-driven default persona
        # (Settings.persona_config also honors the PERSONA_PATH env var) — this is
        # the behavior the config docstrings describe.
        from ovos_persona_server.config import Settings
        persona = Settings().persona_config
        persona["name"] = persona.get("name") or "persona"

    # TODO - move to dependency injection
    ovos_persona_server.persona.default_persona = persona = Persona(persona["name"], persona)

    from ovos_persona_server.version import VERSION_MAJOR, VERSION_ALPHA, VERSION_BUILD, VERSION_MINOR

    version_str = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}"
    if VERSION_ALPHA:
        version_str += f"a{VERSION_ALPHA}"

    app = FastAPI(title="OVOS Persona Server",
                  description="OpenAI/Ollama compatible API for OVOS Personas and Solvers",
                  version=version_str,
                  lifespan=_lifespan)

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
    from ovos_persona_server.cohere import cohere_router
    from ovos_persona_server.huggingface_tgi import tgi_router
    from ovos_persona_server.aws_bedrock import bedrock_router
    from ovos_persona_server.gemini import gemini_router
    from ovos_persona_server.anthropic import anthropic_router
    from ovos_persona_server.deprecated_routers import (
        add_deprecation_middleware,
        register_deprecated_routes,
    )

    # Canonical prefixed routers
    app.include_router(chat_router)         # /openai/v1/...
    app.include_router(ollama_router)       # /ollama/api/...
    app.include_router(utcp_router)
    app.include_router(cohere_router)       # /cohere/v1/...
    app.include_router(tgi_router)          # /tgi/...
    app.include_router(bedrock_router)      # /bedrock/model/...
    app.include_router(gemini_router)       # /gemini/v1beta/models/...
    app.include_router(anthropic_router)    # /anthropic/v1/...

    # OpenAI-compatible RAG surface: files and vector stores. The /embeddings
    # endpoint is already served by chat_router (persona-solver backed), so it
    # is not re-mounted here; vector-store search still uses a dedicated
    # embeddings plugin via embeddings.get_text_embeddings. Optional — mounted
    # only when the subsystem imports cleanly.
    try:
        from ovos_persona_server.files import files_router
        from ovos_persona_server.vector_stores import vector_stores_router
        app.include_router(files_router)            # /openai/v1/files/...
        app.include_router(vector_stores_router)    # /openai/v1/vector_stores/...
    except ImportError as exc:
        logging.getLogger(__name__).warning(
            "RAG endpoints not available (%s); install ovos-persona-server[rag]", exc
        )

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
            logging.getLogger(__name__).warning(
                "a2a_base_url was set but a2a-sdk is not installed — "
                "A2A endpoint will not be available. "
                "Install with: uv pip install 'ovos-persona-server[a2a]'"
            )

    # Mount MCP server (streamable-HTTP transport) when the `mcp` package is available.
    # mount_mcp_on_app sets streamable_http_path="/" so the endpoint lands at /mcp
    # (not /mcp/mcp) and chains the session manager into the host app lifespan.
    try:
        from ovos_persona_server.mcp_server import mount_mcp_on_app
        mount_mcp_on_app(app)
    except ImportError:
        pass  # mcp extra not installed — UTCP only

    return app
