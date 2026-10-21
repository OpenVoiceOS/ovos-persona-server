"""Streamed sentences must keep the space ``Persona.chat`` puts between them.

``Persona.stream`` yields whole sentences with no separator. ``run_stream``
adds the space once, so every streaming route emits text that a client can
concatenate: ``"Diana. This"``, never ``"Diana.This"``.
"""
import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ovos_persona_server.persona import get_default_persona, run_stream

SENTENCES = ["She is Diana.", "This is her story."]
EXPECTED = "She is Diana. This is her story."


def _persona(sentences):
    persona = MagicMock()
    persona.name = "spacing-persona"
    persona.memory = None
    persona.stream.return_value = iter(sentences)
    return persona


def _client(router, persona):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_default_persona] = lambda: persona
    return TestClient(app)


@pytest.mark.parametrize("sentences, expected", [
    (SENTENCES, EXPECTED),
    (["She is Diana. ", "This is her story."], EXPECTED),
    (["She is Diana.", " This is her story."], EXPECTED),
    (["One.", "", "Two."], "One. Two."),
    (["One.", "Two.", "Three."], "One. Two. Three."),
    (["你好。", "再见。"], "你好。再见。"),
    (["こんにちは。", "さようなら。"], "こんにちは。さようなら。"),
    (["他说：「好。」", "然后走了。"], "他说：「好。」然后走了。"),
    (["안녕하세요.", "반갑습니다."], "안녕하세요. 반갑습니다."),
    (["你好。", "Hello."], "你好。Hello."),
    (["他说：“我很好。”", "然后他走了。"], "他说：“我很好。”然后他走了。"),
    (["他想了想……", "然后走了。"], "他想了想……然后走了。"),
    (["Welcome to the city 🌆", "The smell is great."], "Welcome to the city 🌆 The smell is great."),
    (["Hello.", "你好。"], "Hello. 你好。"),
])
def test_run_stream_joins_sentences_with_one_space(sentences, expected):
    assert "".join(run_stream(_persona(sentences), [])) == expected


def test_run_stream_with_memory_persists_the_spaced_reply():
    persona = _persona(SENTENCES)
    persona.memory = MagicMock()
    persona.memory.build_conversation_context.return_value = []
    out = "".join(run_stream(persona, [{"role": "user", "content": "who"}],
                             memory=True, session_id="s"))
    assert out == EXPECTED
    saved = persona.memory.update_history.call_args[0][0]
    assert saved[1].content == EXPECTED


def test_openai_chat_stream_deltas_concatenate_with_spaces():
    from ovos_persona_server.chat import chat_router

    resp = _client(chat_router, _persona(SENTENCES)).post(
        "/openai/v1/chat/completions",
        json={"model": "spacing-persona", "stream": True,
              "messages": [{"role": "user", "content": "who"}]})
    events = [json.loads(line[len("data: "):]) for line in resp.text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    text = "".join(c["delta"].get("content") or ""
                   for e in events for c in e.get("choices", []))
    assert text == EXPECTED


def test_tgi_stream_tokens_concatenate_with_spaces():
    from ovos_persona_server.huggingface_tgi import tgi_router

    resp = _client(tgi_router, _persona(SENTENCES)).post(
        "/tgi/generate_stream", json={"inputs": "who"})
    events = [json.loads(line[len("data:"):]) for line in resp.text.splitlines()
              if line.startswith("data:")]
    tokens = "".join(e["token"]["text"] for e in events if not e["token"]["special"])
    assert tokens == EXPECTED
    assert events[-1]["generated_text"] == EXPECTED
