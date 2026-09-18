"""Unit tests for the server-side transparent-memory toggle (CHAT_MEMORY).

``off`` (default) -> stateless backend: client messages pass through untouched.
``transparent``  -> hosted agent: the persona's memory plugin builds context and
the exchange is persisted, keyed by session.
"""
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from ovos_persona_server import persona as persona_mod
from ovos_persona_server.persona import run_chat, run_stream, memory_enabled


class _FakeMemory:
    """Records build/update calls and returns a canned, augmented context."""

    def __init__(self):
        self.built = []          # (utterance, session_id)
        self.persisted = []      # (messages, session_id)
        self.history = {}        # session_id -> list[AgentMessage]

    def build_conversation_context(self, utterance, session_id):
        self.built.append((utterance, session_id))
        ctx = [AgentMessage(MessageRole.SYSTEM, "you are helpful")]
        ctx.extend(self.history.get(session_id, []))
        ctx.append(AgentMessage(MessageRole.USER, utterance))
        return ctx

    def update_history(self, new_messages, session_id):
        self.persisted.append((new_messages, session_id))
        self.history.setdefault(session_id, []).extend(new_messages)


class _FakePersona:
    """Captures whatever was handed to chat/stream so we can assert on it."""

    def __init__(self, memory=None, reply="hello"):
        self.name = "fake"
        self.memory = memory
        self._reply = reply
        self.seen_messages = None

    def chat(self, messages, sess=None):
        self.seen_messages = messages
        return self._reply

    def stream(self, messages, sess=None):
        self.seen_messages = messages
        # real engines emit chunks carrying their own whitespace; split keeping it
        for tok in self._reply.split(" "):
            yield tok + " "


_MSGS = [{"role": "user", "content": "hi there"}]


def test_off_mode_converts_client_messages():
    mem = _FakeMemory()
    p = _FakePersona(memory=mem)
    out = run_chat(p, _MSGS, memory=False)
    assert out == "hello"
    # client dicts were converted to AgentMessage before reaching the engine; memory untouched
    assert len(p.seen_messages) == 1
    assert isinstance(p.seen_messages[0], AgentMessage)
    assert p.seen_messages[0].role == MessageRole.USER
    assert p.seen_messages[0].content == "hi there"
    assert mem.built == [] and mem.persisted == []


def test_transparent_mode_builds_context_and_persists():
    mem = _FakeMemory()
    p = _FakePersona(memory=mem, reply="the answer")
    out = run_chat(p, _MSGS, memory=True, session_id="user-42")
    assert out == "the answer"
    # context came from the memory plugin (system + utterance), not raw dicts
    assert mem.built == [("hi there", "user-42")]
    assert p.seen_messages[0].role == MessageRole.SYSTEM
    assert p.seen_messages[-1].content == "hi there"
    # the exchange was persisted under the same session key
    (persisted, sid), = mem.persisted
    assert sid == "user-42"
    assert [m.role for m in persisted] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert persisted[1].content == "the answer"


def test_transparent_mode_without_memory_plugin_falls_back():
    p = _FakePersona(memory=None)
    out = run_chat(p, _MSGS, memory=True)
    assert out == "hello"
    # falls back to the stateless (converted) path when no memory plugin is configured
    assert len(p.seen_messages) == 1
    assert isinstance(p.seen_messages[0], AgentMessage)
    assert p.seen_messages[0].content == "hi there"


def test_transparent_mode_is_multi_turn():
    mem = _FakeMemory()
    p = _FakePersona(memory=mem, reply="ok")
    run_chat(p, [{"role": "user", "content": "first"}], memory=True, session_id="s1")
    run_chat(p, [{"role": "user", "content": "second"}], memory=True, session_id="s1")
    # second turn's context must contain the first exchange from server memory
    roles = [m.role for m in p.seen_messages]
    contents = [m.content for m in p.seen_messages]
    assert "first" in contents and "ok" in contents and contents[-1] == "second"
    assert MessageRole.ASSISTANT in roles


def test_stream_transparent_accumulates_and_persists():
    mem = _FakeMemory()
    p = _FakePersona(memory=mem, reply="one two three")
    toks = list(run_stream(p, _MSGS, memory=True, session_id="s2"))
    assert "".join(toks).strip() == "one two three"
    (persisted, sid), = mem.persisted
    assert sid == "s2"
    # streamed chunks reassembled verbatim and persisted
    assert persisted[1].content == "".join(toks)


def test_stream_off_mode_converts_client_messages():
    mem = _FakeMemory()
    p = _FakePersona(memory=mem, reply="a b")
    toks = list(run_stream(p, _MSGS, memory=False))
    assert "".join(toks).strip() == "a b"
    assert len(p.seen_messages) == 1
    assert isinstance(p.seen_messages[0], AgentMessage)
    assert p.seen_messages[0].content == "hi there"
    assert mem.persisted == []


def test_stateless_chat_and_stream_deliver_agent_messages():
    """Regression for PR #67: the stateless path must hand persona.chat/stream
    real AgentMessage objects, with legacy role mapping and content-parts
    flattening applied, not raw request dicts.
    """
    msgs = [
        {"role": "system", "content": "be nice"},
        {"role": "function", "content": "42", "name": "calc"},
        {"role": "bogus", "content": "???"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]},
    ]

    chat_persona = _FakePersona()
    reply = run_chat(chat_persona, msgs, memory=False)
    assert reply == "hello"
    seen = chat_persona.seen_messages
    assert all(isinstance(m, AgentMessage) for m in seen)
    assert seen[1].role == MessageRole.TOOL  # legacy "function" -> tool
    assert seen[2].role == MessageRole.USER  # unknown role falls back to user
    assert seen[-1].content == "hi there"  # content-parts flattened to a string

    stream_persona = _FakePersona(reply="a b")
    list(run_stream(stream_persona, msgs, memory=False))
    seen_stream = stream_persona.seen_messages
    assert all(isinstance(m, AgentMessage) for m in seen_stream)
    assert seen_stream[1].role == MessageRole.TOOL
    assert seen_stream[2].role == MessageRole.USER
    assert seen_stream[-1].content == "hi there"


def test_default_toggle_reads_settings(monkeypatch):
    monkeypatch.setattr(persona_mod.settings, "chat_memory", "off")
    assert memory_enabled() is False
    monkeypatch.setattr(persona_mod.settings, "chat_memory", "transparent")
    assert memory_enabled() is True
    # default param path picks up the setting
    mem = _FakeMemory()
    p = _FakePersona(memory=mem)
    run_chat(p, _MSGS)  # memory=None -> resolves from settings (transparent)
    assert mem.built and mem.persisted
