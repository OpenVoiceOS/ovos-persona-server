"""OpenAI collection routes answer at their documented path, without a trailing slash.

``POST /v1/files`` and ``POST /v1/vector_stores`` are the paths the OpenAI API
publishes. A 307 to ``.../files/`` only works for clients that follow
redirects; curl, httpx and most HTTP libraries do not by default.
"""
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ovos_persona_server.files import files_router
from ovos_persona_server.metadata import init_db
from ovos_persona_server.vector_stores import vector_stores_router


@asynccontextmanager
async def _tables(app: FastAPI):
    await init_db()
    yield


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = FastAPI(lifespan=_tables)
    app.include_router(files_router)
    app.include_router(vector_stores_router)
    with TestClient(app, follow_redirects=False) as c:
        yield c


@pytest.mark.parametrize("path", ["/openai/v1/files", "/openai/v1/vector_stores"])
def test_collection_get_has_no_redirect(client, path):
    r = client.get(path)
    assert r.status_code == 200, (r.status_code, r.headers.get("location"), r.text[:200])
    assert r.json()["object"] == "list"


def test_file_upload_without_trailing_slash(client):
    r = client.post("/openai/v1/files", files={"file": ("a.txt", b"hello")}, data={"purpose": "assistants"})
    assert r.status_code == 200, (r.status_code, r.headers.get("location"), r.text[:200])
    assert r.json()["filename"] == "a.txt"
