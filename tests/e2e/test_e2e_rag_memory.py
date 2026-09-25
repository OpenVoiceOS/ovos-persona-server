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
"""Cross-repo end-to-end test: the ovos-openai-plugin RAG *memory* plugin.

RAG is modelled as an OVOS persona memory plugin (``AgentContextManager``), not a
solver: ``ovos-openai-rag-memory-plugin`` (``PersonaServerRAGMemory``) searches a
vector store hosted by a live persona-server and injects the retrieved chunks into
the conversation context. The persona's normal chat engine would then generate the
answer with that context.

This drives the genuine deploy stack:

- server text embedder: ``ovos-gguf-embeddings-plugin`` (all-MiniLM-L6-v2)
- server vector DB: ``ovos-chromadb-embeddings-plugin``
- client: a real ``ovos_persona.Persona`` whose ``memory_module`` is the RAG memory
  plugin, configured from the persona JSON (exercises ovos-persona passing config
  through to the memory plugin)

Asserting on the context the persona builds proves the memory plugin retrieved the
right chunk through the new ``/vector_stores/{id}/search`` endpoint.

Run::

    pytest tests/e2e/test_e2e_rag_memory.py -v
"""
from __future__ import annotations

import importlib
import json
import os
import socket
import tempfile
import threading
import time
from typing import Generator

import httpx
import pytest
import uvicorn

import openai

# Plugins / packages that must be importable (fail loudly if absent).
from ovos_gguf_plugin.embeddings import GGUFEmbeddings  # noqa: F401
from ovos_chromadb_embeddings import ChromaEmbeddingsDB  # noqa: F401
from ovos_persona import Persona
from ovos_bus_client.session import Session
from ovos_plugin_manager.templates.agents import MessageRole
from ovos_openai_plugin.rag_memory import PersonaServerRAGMemory  # noqa: F401

# Server persona is irrelevant to this test (only its /files + /vector_stores are
# used), so the lightweight failure solver keeps startup cheap.
_SERVER_PERSONA = {"name": "Failer", "solvers": ["ovos-solver-failure-plugin"]}

_DOCS = {
    "cats.txt": b"the cat sat on the mat. cats are fluffy animals that purr.",
    "python.txt": b"python is a programming language used for data science and the web.",
    "moon.txt": b"the moon orbits the earth and affects ocean tides.",
}
_QUERY = "what fluffy animal sits on a mat?"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app(storage_path: str):
    os.environ["TEXT_EMBEDDINGS_PLUGIN"] = "ovos-gguf-embeddings-plugin"
    os.environ["EMBEDDINGS_MODEL"] = "all-MiniLM-L6-v2"
    os.environ["EMBEDDINGS_DB_PLUGIN"] = "ovos-chromadb-embeddings-plugin"
    os.environ["FILE_STORAGE_PATH"] = storage_path

    import ovos_persona_server.config as _cfg_mod
    importlib.reload(_cfg_mod)
    import ovos_persona_server.embeddings as _emb_mod
    import ovos_persona_server.vector_stores as _vs_mod
    importlib.reload(_emb_mod)
    importlib.reload(_vs_mod)

    from ovos_persona_server import create_persona_app

    persona_file = os.path.join(storage_path, "persona.json")
    with open(persona_file, "w") as f:
        json.dump(_SERVER_PERSONA, f)
    return create_persona_app(persona_file)


@pytest.fixture(scope="module")
def live_server() -> Generator[str, None, None]:
    storage_path = tempfile.mkdtemp(prefix="persona-rag-memory-e2e-")
    port = _free_port()
    app = _build_app(storage_path)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60  # first run downloads the embedding model
    while time.time() < deadline:
        try:
            httpx.get(f"{base}/openai/v1/files/", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        server.should_exit = True
        raise RuntimeError("server did not start in time")

    yield base

    server.should_exit = True
    thread.join(timeout=10)


def _populated_vector_store(base: str) -> str:
    client = openai.OpenAI(base_url=f"{base}/openai/v1", api_key="unused")
    file_ids = [client.files.create(file=(name, body), purpose="assistants").id
                for name, body in _DOCS.items()]
    vs = client.vector_stores.create(name="rag-memory-e2e")
    for fid in file_ids:
        client.vector_stores.files.create(vector_store_id=vs.id, file_id=fid)

    deadline = time.time() + 30
    while time.time() < deadline:
        statuses = {f.status for f in client.vector_stores.files.list(vector_store_id=vs.id).data}
        if statuses <= {"completed"}:
            break
        time.sleep(0.5)
    else:
        pytest.fail(f"files did not embed in time; statuses={statuses}")
    return vs.id


def _has_cat_context(text: str) -> bool:
    text = text.lower()
    return "cat" in text or "fluffy" in text or "mat" in text


@pytest.mark.parametrize("inject_mode", ["system", "tool"])
def test_rag_memory_augments_persona_context(live_server, inject_mode):
    """A persona with the RAG memory plugin retrieves real context from the server.

    Proves the cross-repo path: Persona (memory_module=ovos-openai-rag-memory-plugin,
    configured via persona JSON) -> /vector_stores/{id}/search (gguf + chromadb) ->
    retrieved chunk injected into the context the chat engine would receive. Run for
    both the default ``system`` injection and the ``tool`` (tool-call result) mode.
    """
    base = live_server
    vector_store_id = _populated_vector_store(base)

    persona_cfg = {
        "name": "kb-assistant",
        "solvers": ["ovos-solver-failure-plugin"],
        "memory_module": "ovos-openai-rag-memory-plugin",
        "ovos-openai-rag-memory-plugin": {
            "api_url": f"{base}/openai/v1",
            "vector_store_id": vector_store_id,
            "system_prompt": "You are a helpful assistant.",
            "inject_mode": inject_mode,
            "retrieval": {"max_num_results": 3},
        },
    }
    persona = Persona("kb-assistant", persona_cfg)

    # The persona must have loaded our memory plugin WITH its config (ovos-persona
    # config-passing) — not the default short-term memory.
    assert isinstance(persona.memory, PersonaServerRAGMemory), (
        f"expected RAG memory plugin, got {type(persona.memory).__name__}"
    )

    # Building context triggers the live vector-store search and context injection.
    messages = persona.get_messages(_QUERY, Session())

    assert messages, "persona built no context messages"
    # context-manager contract: last message is the current user utterance
    assert messages[-1].role == MessageRole.USER
    assert messages[-1].content.strip() == _QUERY

    if inject_mode == "tool":
        # retrieved context arrives as a TOOL result paired with an assistant tool_call
        asst = [m for m in messages if m.role == MessageRole.ASSISTANT and m.tool_calls]
        tool = [m for m in messages if m.role == MessageRole.TOOL]
        assert asst and tool, "tool mode did not inject a tool-call/result pair"
        assert tool[0].tool_call_id == asst[0].tool_calls[0].id
        assert _has_cat_context(tool[0].content), (
            f"retrieved cat context not in tool result: {tool[0].content!r}"
        )
    else:
        system_text = " ".join(m.content for m in messages if m.role == MessageRole.SYSTEM)
        assert system_text, "no system message — retrieved context was not injected"
        assert _has_cat_context(system_text), (
            f"retrieved cat context not present in injected system prompt: {system_text!r}"
        )
