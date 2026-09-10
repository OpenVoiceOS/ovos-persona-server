"""An HTTP failure from the persona's backend is a gateway error, not a server bug.

A persona whose chat engine talks to a remote model may get a 401, 429 or
5xx back. The route reports that as 502 and names the upstream status, so a
client can tell a bad backend key from a crash in this server.
"""
from unittest.mock import MagicMock

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

import ovos_persona_server.chat as chat_mod
from ovos_persona_server import persona as persona_mod


def _upstream_error(status_code):
    response = MagicMock()
    response.status_code = status_code
    http_err = requests.HTTPError(f"{status_code} Client Error", response=response)
    try:
        raise requests.RequestException(f"HTTP error: {http_err}") from http_err
    except requests.RequestException as e:
        return e


class _FailingPersona:
    name = "bad"
    memory = None

    class solvers:
        modules = []
        loaded_modules = {}

    def __init__(self, exc):
        self._exc = exc

    def chat(self, messages, sess=None, **kwargs):
        raise self._exc

    def stream(self, messages, sess=None, **kwargs):
        raise self._exc


def _client(exc):
    app = FastAPI()
    app.include_router(chat_mod.chat_router)
    app.dependency_overrides[persona_mod.get_default_persona] = lambda: _FailingPersona(exc)
    return TestClient(app)


@pytest.mark.parametrize("code", [401, 429, 503])
def test_upstream_http_error_is_502_with_status(code):
    r = _client(_upstream_error(code)).post(
        "/openai/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502, r.text
    assert str(code) in r.json()["detail"]


def test_plain_exception_stays_500():
    r = _client(RuntimeError("boom")).post(
        "/openai/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 500
