"""
Main application entry point for the OVOS Persona Server.

This module initializes the FastAPI application, sets up CORS middleware,
and includes various API routers for chat, embeddings, Ollama, persona status,
and mock OpenAI Vector Stores. It now centrally manages the unified SQLite database
initialization using SQLAlchemy.
"""
import glob
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ovos_persona import Persona

import ovos_persona_server.persona
from ovos_persona_server.persona import ModelNotFoundError, register_personas


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


def _load_persona(path: str) -> Persona:
    """Build a :class:`Persona` from a persona JSON file.

    The persona's ``name`` is its model id on every API surface. When the file
    does not set one, the file name is used (unchanged legacy behaviour).
    """
    with open(path) as f:
        config = json.load(f)
    config["name"] = config.get("name") or os.path.basename(path)
    return Persona(config["name"], config)


def _collect_persona_paths(persona_path: Optional[str] = None,
                           personas_dir: Optional[str] = None) -> List[str]:
    """Resolve the CLI persona selectors to an ordered list of JSON files.

    ``persona_path`` accepts a single path (legacy) or a sequence of paths;
    ``personas_dir`` contributes every ``*.json`` in the directory, sorted by
    name so the load order is stable.
    """
    paths: List[str] = []
    if persona_path:
        if isinstance(persona_path, (str, bytes, os.PathLike)):
            paths.append(str(persona_path))
        else:
            paths.extend(str(p) for p in persona_path)
    if personas_dir:
        if not os.path.isdir(personas_dir):
            raise ValueError(f"--personas-dir is not a directory: {personas_dir}")
        found = sorted(glob.glob(os.path.join(personas_dir, "*.json")))
        if not found:
            raise ValueError(f"no *.json persona files found in {personas_dir}")
        paths.extend(p for p in found if p not in paths)
    return paths


def create_persona_app(persona_path: Optional[str] = None,
                       a2a_base_url: Optional[str] = None,
                       personas_dir: Optional[str] = None,
                       default_persona: Optional[str] = None,
                       enable_mcp: bool = False) -> FastAPI:
    """
    Creates and configures the FastAPI application for the Persona Server.

    One process can serve several personas. A persona's ``name`` is its model
    id: clients pick a persona through the ``model`` field (or the model path
    segment) of whichever vendor API they speak. See ``docs/multi-persona.md``.

    Args:
        persona_path: Path to a persona JSON file, or a sequence of paths.
        a2a_base_url: If provided, mounts an A2A-compatible endpoint at ``/a2a``
                      using this URL as the public base URL in the Agent Card
                      (e.g. ``http://myhost:8337/a2a``). Each persona also gets
                      its own card at ``/a2a/<persona name>``. Requires
                      ``a2a-sdk`` (``uv pip install 'ovos-persona-server[a2a]'``).
        personas_dir: Directory whose ``*.json`` files are all loaded as personas.
        default_persona: Name of the persona that answers requests which do not
                         name a model. Defaults to the first persona loaded.
        enable_mcp: Mount the MCP server (streamable-HTTP transport) at ``/mcp``.
                    Opt-in: installing the ``mcp`` extra alone does not expose
                    the endpoint. Requires ``ovos-persona-server[mcp]``.

    Returns:
        FastAPI: The configured FastAPI application instance.
    """
    paths = _collect_persona_paths(persona_path, personas_dir)
    if paths:
        loaded = [_load_persona(p) for p in paths]
    else:
        # No persona file: fall back to the env-var-driven default persona
        # (Settings.persona_config also honors the PERSONA_PATH env var) — this is
        # the behavior the config docstrings describe.
        from ovos_persona_server.config import Settings
        config = Settings().persona_config
        config["name"] = config.get("name") or "persona"
        loaded = [Persona(config["name"], config)]

    # TODO - move to dependency injection
    persona = register_personas(loaded, default_persona)

    from ovos_persona_server.version import VERSION_MAJOR, VERSION_ALPHA, VERSION_BUILD, VERSION_MINOR

    version_str = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}"
    if VERSION_ALPHA:
        version_str += f"a{VERSION_ALPHA}"

    app = FastAPI(title="OVOS Persona Server",
                  description="OpenAI/Ollama compatible API for OVOS Personas and Solvers",
                  version=version_str,
                  lifespan=_lifespan)

    @app.exception_handler(ModelNotFoundError)
    async def _model_not_found(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        """Render an unknown model id as an OpenAI-shaped error object."""
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

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
            # Per-persona agent cards first: Starlette matches mounts in order,
            # so "/a2a" would otherwise swallow "/a2a/<name>/...".
            for name, p in ovos_persona_server.persona.personas.items():
                app.mount(f"/a2a/{name}",
                          create_a2a_application(p, f"{a2a_base_url.rstrip('/')}/{name}"))
            a2a_starlette = create_a2a_application(persona, a2a_base_url)
            app.mount("/a2a", a2a_starlette)
        else:
            logging.getLogger(__name__).warning(
                "a2a_base_url was set but a2a-sdk is not installed — "
                "A2A endpoint will not be available. "
                "Install with: uv pip install 'ovos-persona-server[a2a]'"
            )

    # Mount MCP server (streamable-HTTP transport) — opt-in via enable_mcp/--mcp.
    # mount_mcp_on_app sets streamable_http_path="/" so the endpoint lands at /mcp
    # (not /mcp/mcp) and chains the session manager into the host app lifespan.
    if enable_mcp:
        try:
            from ovos_persona_server.mcp_server import mount_mcp_on_app
            mount_mcp_on_app(app)
        except ImportError:
            logging.getLogger(__name__).warning(
                "--mcp was set but the mcp extra is not installed — "
                "the /mcp endpoint will not be available. "
                "Install with: uv pip install 'ovos-persona-server[mcp]'"
            )

    return app
