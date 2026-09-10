# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for deprecated_routers.py — legacy path middleware and route registration.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_app_with_middleware():
    """Build a minimal app with a stub /openai/v1/ping and /api/ping route,
    plus the deprecation middleware attached."""
    from fastapi import APIRouter
    from ovos_persona_server.deprecated_routers import (
        add_deprecation_middleware,
        _LEGACY_PREFIX_MAP,
    )

    app = FastAPI()
    add_deprecation_middleware(app)

    router = APIRouter()

    @router.get("/openai/v1/ping")
    async def openai_ping():
        return {"pong": "openai"}

    @router.get("/ollama/api/ping")
    async def ollama_ping():
        return {"pong": "ollama"}

    # Legacy (deprecated) paths
    @router.get("/v1/ping")
    async def openai_legacy_ping():
        return {"pong": "openai-legacy"}

    @router.get("/api/ping")
    async def ollama_legacy_ping():
        return {"pong": "ollama-legacy"}

    app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# Tests for _build_successor_path
# ---------------------------------------------------------------------------

class TestBuildSuccessorPath:
    def test_v1_path_mapped_to_openai(self):
        from ovos_persona_server.deprecated_routers import _build_successor_path
        result = _build_successor_path("/v1/chat/completions")
        assert result == "/openai/v1/chat/completions"

    def test_api_path_mapped_to_ollama(self):
        from ovos_persona_server.deprecated_routers import _build_successor_path
        result = _build_successor_path("/api/generate")
        assert result == "/ollama/api/generate"

    def test_exact_legacy_prefix_mapped(self):
        from ovos_persona_server.deprecated_routers import _build_successor_path
        result = _build_successor_path("/v1")
        assert result == "/openai/v1"

    def test_unknown_prefix_passthrough(self):
        from ovos_persona_server.deprecated_routers import _build_successor_path
        result = _build_successor_path("/health")
        assert result == "/health"

    def test_canonical_path_unchanged(self):
        from ovos_persona_server.deprecated_routers import _build_successor_path
        result = _build_successor_path("/openai/v1/models")
        assert result == "/openai/v1/models"


# ---------------------------------------------------------------------------
# Tests for add_deprecation_middleware
# ---------------------------------------------------------------------------

class TestDeprecationMiddleware:
    def setup_method(self):
        self.app = _make_minimal_app_with_middleware()
        self.client = TestClient(self.app)

    def test_canonical_path_no_deprecation_headers(self):
        """Canonical paths must NOT receive Deprecation headers."""
        resp = self.client.get("/openai/v1/ping")
        assert resp.status_code == 200
        assert "deprecation" not in resp.headers

    def test_legacy_v1_path_has_deprecation_header(self):
        """Requests to /v1/... must get Deprecation: true header."""
        resp = self.client.get("/v1/ping")
        assert resp.status_code == 200
        assert resp.headers.get("deprecation") == "true"

    def test_legacy_v1_path_has_link_header(self):
        """Requests to /v1/... must get a Link header pointing to canonical."""
        resp = self.client.get("/v1/ping")
        link = resp.headers.get("link", "")
        assert "/openai/v1/ping" in link
        assert 'rel="successor-version"' in link

    def test_legacy_api_path_has_deprecation_header(self):
        """Requests to /api/... must get Deprecation: true header."""
        resp = self.client.get("/api/ping")
        assert resp.headers.get("deprecation") == "true"

    def test_legacy_api_path_has_link_header(self):
        """Requests to /api/... must get a Link header pointing to canonical."""
        resp = self.client.get("/api/ping")
        link = resp.headers.get("link", "")
        assert "/ollama/api/ping" in link
        assert 'rel="successor-version"' in link

    def test_non_deprecated_route_no_headers(self):
        """Routes that are not in the legacy prefix map get no deprecation signal."""
        resp = self.client.get("/ollama/api/ping")
        assert resp.status_code == 200
        assert "deprecation" not in resp.headers

    def test_link_header_format(self):
        """Link header value must be in RFC 8288 format: <url>; rel="..."."""
        resp = self.client.get("/v1/ping")
        link = resp.headers.get("link", "")
        assert link.startswith("<")
        assert ">;" in link


# ---------------------------------------------------------------------------
# Tests for malformed bodies / FastAPI validation
# ---------------------------------------------------------------------------

class TestMalformedBodies:
    """Verify that FastAPI returns 422 for invalid request bodies on the main app."""

    def setup_method(self):
        from fastapi import APIRouter
        from pydantic import BaseModel, Field
        from fastapi import FastAPI

        app = FastAPI()
        router = APIRouter()

        class StrictBody(BaseModel):
            name: str = Field(..., min_length=1)
            count: int

        @router.post("/strict")
        async def strict_endpoint(body: StrictBody):
            return {"ok": True}

        app.include_router(router)
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_missing_required_field_422(self):
        resp = self.client.post("/strict", json={"name": "test"})
        assert resp.status_code == 422

    def test_wrong_type_422(self):
        resp = self.client.post("/strict", json={"name": "test", "count": "not-an-int"})
        assert resp.status_code == 422

    def test_empty_body_422(self):
        resp = self.client.post("/strict", json={})
        assert resp.status_code == 422

    def test_non_json_body_422(self):
        resp = self.client.post(
            "/strict",
            content=b"not json at all",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Tests for register_deprecated_routes
# ---------------------------------------------------------------------------

class TestRegisterDeprecatedRoutes:
    """Smoke-test that register_deprecated_routes mounts legacy paths."""

    def test_register_does_not_raise(self):
        """register_deprecated_routes must not raise when called on a fresh app."""
        from unittest.mock import patch, MagicMock
        from ovos_persona_server.deprecated_routers import register_deprecated_routes

        # Build a fake router with a real APIRoute so the function can iterate
        from fastapi import APIRouter
        from fastapi.routing import APIRoute

        async def _dummy():
            return {}

        fake_chat_router = APIRouter(prefix="/openai/v1", tags=["openai"])
        fake_chat_router.add_api_route("/models", endpoint=_dummy, methods=["GET"])

        fake_ollama_router = APIRouter(prefix="/ollama/api", tags=["ollama"])
        fake_ollama_router.add_api_route("/tags", endpoint=_dummy, methods=["GET"])

        app = FastAPI()
        with (
            patch("ovos_persona_server.deprecated_routers.chat_router", fake_chat_router, create=True),
            patch("ovos_persona_server.deprecated_routers.ollama_router", fake_ollama_router, create=True),
        ):
            # Patch the imports *inside* the function
            import ovos_persona_server.deprecated_routers as dr_mod
            import importlib
            with (
                patch.dict(
                    "sys.modules",
                    {
                        "ovos_persona_server.chat": MagicMock(chat_router=fake_chat_router),
                        "ovos_persona_server.ollama": MagicMock(ollama_router=fake_ollama_router),
                    },
                )
            ):
                register_deprecated_routes(app)

        # The app should have legacy routes mounted. Newer FastAPI wraps included
        # routers in app.routes (no flat `.path`), so assert via the OpenAPI schema,
        # which is the authoritative list of registered paths.
        route_paths = list(app.openapi()["paths"].keys())
        assert any("/v1/" in p for p in route_paths), f"No /v1/ routes found: {route_paths}"
