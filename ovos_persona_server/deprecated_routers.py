# Licensed under the Apache License, Version 2.0
"""Deprecated legacy routers for OpenAI and Ollama endpoints.

The canonical prefixed paths are:
  /openai/v1/...  (OpenAI-compatible)
  /ollama/api/... (Ollama-compatible)

``register_deprecated_routes()`` mounts the same router instances a second time
at the legacy unprefixed paths:
  /v1/...
  /api/...

A FastAPI middleware injects response headers on any request whose path starts
with a legacy prefix:
  Deprecation: true
  Link: <canonical_path>; rel="successor-version"

This preserves full FastAPI dependency injection (``Depends``) on every route
while still signalling clients to migrate.
"""
import logging
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRoute

LOG = logging.getLogger(__name__)

# Mapping from legacy prefix → canonical prefix
_LEGACY_PREFIX_MAP = {
    "/v1": "/openai/v1",
    "/api": "/ollama/api",
}


def _build_successor_path(request_path: str) -> str:
    """Compute the canonical path for a legacy request path.

    Args:
        request_path: Incoming request URL path starting with a legacy prefix.

    Returns:
        The canonical path with the vendor prefix substituted in, or the
        original path if no mapping is found.
    """
    for legacy, canonical in _LEGACY_PREFIX_MAP.items():
        if request_path.startswith(legacy + "/") or request_path == legacy:
            return canonical + request_path[len(legacy):]
    return request_path


def add_deprecation_middleware(app: FastAPI) -> None:
    """Attach middleware to ``app`` that injects deprecation headers on legacy paths.

    Any response for a path starting with a known legacy prefix receives:
      ``Deprecation: true``
      ``Link: <canonical>; rel="successor-version"``

    A WARNING is also logged on the first hit.

    Args:
        app: The FastAPI application to attach the middleware to.
    """
    @app.middleware("http")
    async def deprecation_header_middleware(request: Request, call_next: Callable) -> Response:
        """Inject Deprecation + Link headers for legacy endpoint paths."""
        path = request.url.path
        is_legacy = any(
            path.startswith(prefix + "/") or path == prefix
            for prefix in _LEGACY_PREFIX_MAP
        )
        response: Response = await call_next(request)
        if is_legacy:
            successor = _build_successor_path(path)
            LOG.warning(
                "Deprecated path accessed: %s — use %s instead",
                path,
                successor,
            )
            response.headers["Deprecation"] = "true"
            response.headers["Link"] = f'<{successor}>; rel="successor-version"'
        return response


def register_deprecated_routes(app: FastAPI) -> None:
    """Mount legacy /v1/... and /api/... routes alongside the canonical prefixed ones.

    Imports chat_router and ollama_router and includes them a second time with
    the legacy prefixes, so existing clients continue to work while receiving
    deprecation signal via response headers (added by the middleware).

    Call ``add_deprecation_middleware(app)`` separately (before the first
    request) to attach the header-injection middleware.

    Args:
        app: The FastAPI application to register the deprecated routes on.
    """
    from ovos_persona_server.chat import chat_router
    from ovos_persona_server.ollama import ollama_router

    # chat_router has prefix="/openai/v1"; re-mount it at "/v1" by passing
    # the routes directly under a new router with the legacy prefix.
    # FastAPI's include_router doesn't allow overriding an existing prefix, so
    # we build a bare router whose route paths start with the legacy prefix.
    from fastapi import APIRouter
    from fastapi.routing import APIRoute as _APIRoute

    def _legacy_router(source_router, new_prefix: str, canonical_prefix: str) -> APIRouter:
        """Clone routes from source_router at new_prefix instead of canonical_prefix."""
        legacy = APIRouter(tags=[f"{source_router.tags[0] if source_router.tags else 'api'} (deprecated — use {canonical_prefix})"])
        for route in source_router.routes:
            if not isinstance(route, _APIRoute):
                continue
            # route.path already contains the canonical prefix, e.g.
            # '/openai/v1/chat/completions'.  Replace it with the legacy prefix.
            relative = route.path[len(canonical_prefix):]   # '/chat/completions'
            legacy_path = new_prefix + relative              # '/v1/chat/completions'
            legacy.add_api_route(
                path=legacy_path,
                endpoint=route.endpoint,
                methods=list(route.methods or ["GET"]),
                response_model=route.response_model,
                status_code=route.status_code,
                dependencies=route.dependencies,
                summary=f"[DEPRECATED] {route.summary or route.name}",
                include_in_schema=True,
                deprecated=True,
            )
        return legacy

    app.include_router(
        _legacy_router(chat_router, new_prefix="/v1", canonical_prefix="/openai/v1")
    )
    app.include_router(
        _legacy_router(ollama_router, new_prefix="/api", canonical_prefix="/ollama/api")
    )
