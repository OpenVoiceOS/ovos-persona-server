"""Tests for serving several personas from one process.

A persona's ``name`` is its model id. Clients select a persona with the
``model`` field (or the model path segment) of whichever vendor API they speak.
The single-persona deployment must keep working exactly as before.
"""
import json
from typing import Dict, Generator, List

import pytest
from fastapi.testclient import TestClient

import ovos_persona_server
from ovos_persona_server import create_persona_app
from ovos_persona_server import persona as persona_mod


class FakePersona:
    """Minimal stand-in for ovos_persona.Persona with its own memory."""

    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.memory = None
        self.seen: List = []

        class _Solvers:
            loaded_modules: dict = {}

        self.solvers = _Solvers()

    def chat(self, messages, sess=None, **kwargs) -> str:
        self.seen.append(messages)
        return f"hello from {self.name}"

    def stream(self, messages, sess=None, **kwargs) -> Generator[str, None, None]:
        self.seen.append(messages)
        yield f"hello from {self.name}"


class RecordingMemory:
    """Memory plugin double that records the session keys it is asked about."""

    def __init__(self):
        self.history: Dict[str, list] = {}
        self.keys: List[str] = []

    def build_conversation_context(self, utterance, session_id):
        self.keys.append(session_id)
        return list(self.history.get(session_id, [])) + [utterance]

    def update_history(self, new_messages, session_id):
        self.keys.append(session_id)
        self.history.setdefault(session_id, []).extend(new_messages)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the process-wide persona registry out of other test modules."""
    saved = dict(persona_mod.personas)
    saved_default = persona_mod.default_persona
    yield
    persona_mod.personas.clear()
    persona_mod.personas.update(saved)
    persona_mod.default_persona = saved_default


@pytest.fixture()
def fake_persona_class(monkeypatch):
    """Make create_persona_app build FakePersona instead of a real Persona."""
    monkeypatch.setattr(ovos_persona_server, "Persona", FakePersona)
    return FakePersona


def _write_persona(tmp_path, name: str) -> str:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"name": name, "solvers": ["fake"]}))
    return str(path)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def test_repeated_persona_flag_loads_all(tmp_path, fake_persona_class):
    a = _write_persona(tmp_path, "alpha")
    b = _write_persona(tmp_path, "beta")
    create_persona_app([a, b])
    assert list(persona_mod.personas) == ["alpha", "beta"]
    assert persona_mod.default_persona.name == "alpha"


def test_personas_dir_loads_every_json(tmp_path, fake_persona_class):
    _write_persona(tmp_path, "beta")
    _write_persona(tmp_path, "alpha")
    create_persona_app(personas_dir=str(tmp_path))
    # sorted by file name so the load order (and thus the default) is stable
    assert list(persona_mod.personas) == ["alpha", "beta"]
    assert persona_mod.default_persona.name == "alpha"


def test_explicit_default_persona(tmp_path, fake_persona_class):
    a = _write_persona(tmp_path, "alpha")
    b = _write_persona(tmp_path, "beta")
    create_persona_app([a, b], default_persona="beta")
    assert persona_mod.default_persona.name == "beta"


def test_unknown_default_persona_is_rejected(tmp_path, fake_persona_class):
    a = _write_persona(tmp_path, "alpha")
    with pytest.raises(ValueError):
        create_persona_app([a], default_persona="nope")


def test_duplicate_persona_names_are_rejected(tmp_path, fake_persona_class):
    a = _write_persona(tmp_path, "alpha")
    other = tmp_path / "copy.json"
    other.write_text(json.dumps({"name": "alpha", "solvers": ["fake"]}))
    with pytest.raises(ValueError):
        create_persona_app([a, str(other)])


def test_single_persona_path_still_loads(tmp_path, fake_persona_class):
    """Backward compatibility: --persona <one file> as a bare string."""
    a = _write_persona(tmp_path, "alpha")
    create_persona_app(a)
    assert list(persona_mod.personas) == ["alpha"]
    assert persona_mod.multi_persona() is False


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

@pytest.fixture()
def multi_client(tmp_path, fake_persona_class) -> TestClient:
    a = _write_persona(tmp_path, "alpha")
    b = _write_persona(tmp_path, "beta")
    return TestClient(create_persona_app([a, b]))


@pytest.fixture()
def single_client(tmp_path, fake_persona_class) -> TestClient:
    a = _write_persona(tmp_path, "alpha")
    return TestClient(create_persona_app(a))


def _content(resp) -> str:
    return resp.json()["choices"][0]["message"]["content"]


def test_models_lists_every_persona(multi_client):
    data = multi_client.get("/openai/v1/models").json()
    assert [m["id"] for m in data["data"]] == ["alpha", "beta"]
    for entry in data["data"]:
        assert entry["object"] == "model"
        assert entry["owned_by"] == "ovos"
        assert isinstance(entry["created"], int)


def test_chat_routes_on_model_name(multi_client):
    for name in ("alpha", "beta"):
        resp = multi_client.post("/openai/v1/chat/completions",
                                 json={"model": name,
                                       "messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 200
        assert _content(resp) == f"hello from {name}"
        assert resp.json()["model"] == name


def test_chat_without_model_uses_default(multi_client):
    resp = multi_client.post("/openai/v1/chat/completions",
                             json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert _content(resp) == "hello from alpha"


def test_unknown_model_is_404_listing_available_names(multi_client):
    resp = multi_client.post("/openai/v1/chat/completions",
                             json={"model": "gpt-4",
                                   "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "model_not_found"
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "model"
    assert "gpt-4" in error["message"]
    assert "alpha" in error["message"] and "beta" in error["message"]


def test_single_persona_ignores_unknown_model(single_client):
    """Backward compatibility: one persona keeps answering whatever model is asked."""
    resp = single_client.post("/openai/v1/chat/completions",
                              json={"model": "gpt-4",
                                    "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert _content(resp) == "hello from alpha"


def test_legacy_completions_route_on_model(multi_client):
    resp = multi_client.post("/openai/v1/completions",
                             json={"model": "beta", "prompt": "hi"})
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["text"] == "hello from beta"


def test_ollama_routes_and_lists(multi_client):
    tags = multi_client.get("/ollama/api/tags").json()
    assert [m["name"] for m in tags["models"]] == ["alpha", "beta"]
    running = multi_client.get("/ollama/api/ps").json()
    assert [m["name"] for m in running["models"]] == ["alpha", "beta"]
    assert multi_client.get("/ollama/api/show", params={"model": "beta"}).json()["name"] == "beta"

    resp = multi_client.post("/ollama/api/chat",
                             json={"model": "beta", "stream": False,
                                   "messages": [{"role": "user", "content": "hi"}]})
    assert resp.json()["message"]["content"] == "hello from beta"
    assert multi_client.post("/ollama/api/chat",
                             json={"model": "nope", "stream": False,
                                   "messages": [{"role": "user", "content": "hi"}]}).status_code == 404


def test_anthropic_routes_on_model(multi_client):
    resp = multi_client.post("/anthropic/v1/messages",
                             json={"model": "beta", "max_tokens": 10,
                                   "messages": [{"role": "user", "content": "hi"}]})
    assert resp.json()["content"][0]["text"] == "hello from beta"
    assert multi_client.post("/anthropic/v1/messages",
                             json={"model": "claude-3-5-sonnet", "max_tokens": 10,
                                   "messages": [{"role": "user", "content": "hi"}]}
                             ).status_code == 404


def test_cohere_routes_on_model_and_defaults(multi_client):
    resp = multi_client.post("/cohere/v1/chat", json={"model": "beta", "message": "hi"})
    assert resp.json()["text"] == "hello from beta"
    # model is optional in the Cohere API — omitting it selects the default
    resp = multi_client.post("/cohere/v1/chat", json={"message": "hi"})
    assert resp.json()["text"] == "hello from alpha"


def test_gemini_routes_on_path_model(multi_client):
    resp = multi_client.post("/gemini/v1beta/models/beta:generateContent",
                             json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    assert resp.json()["candidates"][0]["content"]["parts"][0]["text"] == "hello from beta"
    assert multi_client.post("/gemini/v1beta/models/gemini-pro:generateContent",
                             json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
                             ).status_code == 404


def test_bedrock_model_id_is_permissive(multi_client):
    """The Bedrock model id also picks the response shape, so a vendor id is
    not an error — it falls back to the default persona."""
    resp = multi_client.post("/bedrock/model/beta/invoke",
                             json={"messages": [{"role": "user", "content": "hi"}]})
    assert "hello from beta" in resp.text
    resp = multi_client.post("/bedrock/model/amazon.titan-text-express-v1/invoke",
                             json={"inputText": "hi"})
    assert resp.status_code == 200
    assert "hello from alpha" in resp.text


def test_tgi_always_serves_the_default_persona(multi_client):
    """TGI has no model field — documented as default-persona only."""
    assert multi_client.get("/tgi/info").json()["model_id"] == "alpha"
    resp = multi_client.post("/tgi/generate", json={"inputs": "hi"})
    assert resp.json()["generated_text"] == "hello from alpha"


def test_a2a_mounts_one_card_per_persona(tmp_path, fake_persona_class, monkeypatch):
    from starlette.applications import Starlette

    from ovos_persona_server import a2a as a2a_mod

    class _Builder:
        def __init__(self, persona, base_url):
            self.persona, self.base_url = persona, base_url

        def build(self):
            return Starlette()

    monkeypatch.setattr(a2a_mod, "_A2A_AVAILABLE", True)
    monkeypatch.setattr(a2a_mod, "create_a2a_application", _Builder, raising=False)

    a = _write_persona(tmp_path, "alpha")
    b = _write_persona(tmp_path, "beta")
    app = create_persona_app([a, b], a2a_base_url="http://host:8337/a2a")
    mounted = [r.path for r in app.routes if getattr(r, "path", "").startswith("/a2a")]
    assert "/a2a/alpha" in mounted and "/a2a/beta" in mounted and "/a2a" in mounted
    # the catch-all default mount must come last or it swallows the named ones
    assert mounted.index("/a2a") == len(mounted) - 1


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------

def test_memory_keys_are_namespaced_per_persona(tmp_path, fake_persona_class):
    a = _write_persona(tmp_path, "alpha")
    b = _write_persona(tmp_path, "beta")
    create_persona_app([a, b])
    alpha, beta = persona_mod.personas["alpha"], persona_mod.personas["beta"]
    # a memory plugin backed by shared storage keys only by session id
    shared = RecordingMemory()
    alpha.memory = beta.memory = shared

    persona_mod.run_chat(alpha, [{"role": "user", "content": "secret"}],
                         memory=True, session_id="caller-1")
    persona_mod.run_chat(beta, [{"role": "user", "content": "other"}],
                         memory=True, session_id="caller-1")

    assert set(shared.history) == {"alpha::caller-1", "beta::caller-1"}
    # beta never saw alpha's turn
    assert "secret" not in str(shared.history["beta::caller-1"])


def test_single_persona_memory_keys_are_not_namespaced(tmp_path, fake_persona_class):
    """Backward compatibility: stored conversations stay addressable."""
    a = _write_persona(tmp_path, "alpha")
    create_persona_app(a)
    alpha = persona_mod.personas["alpha"]
    alpha.memory = RecordingMemory()
    persona_mod.run_chat(alpha, [{"role": "user", "content": "hi"}],
                         memory=True, session_id="caller-1")
    assert list(alpha.memory.history) == ["caller-1"]


def test_personas_do_not_share_conversation_state(multi_client):
    """Each persona instance holds its own solvers/memory; nothing crosses over."""
    alpha, beta = persona_mod.personas["alpha"], persona_mod.personas["beta"]
    multi_client.post("/openai/v1/chat/completions",
                      json={"model": "alpha", "messages": [{"role": "user", "content": "secret"}]})
    assert alpha.seen and not beta.seen
    assert "secret" not in str(beta.seen)
