"""
A2A (Agent-to-Agent) server adapter for OVOS Persona Server.

Exposes the loaded persona as an A2A-compatible agent server, allowing any
A2A client to interact with OVOS personas using the standard A2A protocol.

Targets ``a2a-sdk>=1.1.2`` (the ``[http-server]`` extra, which pulls in
``starlette``/``sse-starlette``). The *install* of the 0.3.x line is not
supported — a fresh ``pip install`` cannot reach it, ``a2a-sdk>=0.3.0``
resolves to 1.1.2, whose ``a2a.server.apps`` module was removed in favor of
the route-factory API in ``a2a.server.routes`` used below, and whose
``AgentCard.url`` field was replaced by ``AgentCard.supported_interfaces``.

The *wire protocol* still talks to already-deployed 0.3.x clients: routes
are built with ``enable_v0_3_compat=True`` and the agent card is also
published at the pre-1.0 well-known path, so a remote caller built against
0.3.x is not rejected outright (``-32009``) and can still discover this
agent. If a later change drops that compat flag, existing 0.3.x clients
stop working — that would be a deliberate wire-protocol break, not just an
install-time one, and should be called out as such.

Endpoints (mounted at the path passed to ``create_a2a_application``):
  GET  /.well-known/agent-card.json  — Agent Card (1.x well-known path)
  GET  /.well-known/agent.json       — Agent Card (0.3.x well-known path, compat)
  POST /                              — JSON-RPC 2.0. On 1.x the method name is
                                         ``SendMessage``/``SendStreamingMessage``;
                                         ``enable_v0_3_compat=True`` also accepts
                                         the 0.3.x ``message/send``/``message/stream``
                                         method names from older clients.

A2A spec: https://a2a-protocol.org/
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from ovos_persona_server.persona import run_stream, PersonaNoAnswerError

LOG = logging.getLogger(__name__)

# Sentinel names so monkeypatch works without a2a-sdk installed
AgentCard = None
AgentCapabilities = None
AgentSkill = None
AgentInterface = None
Part = None
TaskState = None
TaskStatus = None
AgentExecutor = object  # base class fallback
EventQueue = None
RequestContext = None
DefaultRequestHandler = None
InMemoryTaskStore = None
TaskUpdater = None
create_agent_card_routes = None
create_jsonrpc_routes = None
Task = None
Starlette = None

try:
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.server.tasks.task_updater import TaskUpdater
    from a2a.types import (
        AgentCapabilities,
        AgentCard,
        AgentInterface,
        AgentSkill,
        Part,
        Task,
        TaskState,
        TaskStatus,
    )
    from starlette.applications import Starlette

    _A2A_AVAILABLE = True
except ImportError:
    _A2A_AVAILABLE = False
    LOG.warning(
        "a2a-sdk is not installed — A2A endpoint will not be available. "
        "Install with: uv pip install 'ovos-persona-server[a2a]'"
    )

if TYPE_CHECKING:
    from ovos_persona import Persona


def _agent_card(persona: "Persona", base_url: str) -> "AgentCard":
    """
    Build an A2A AgentCard from a loaded OVOS persona.

    Args:
        persona: The active ``Persona`` instance.
        base_url: Public base URL of the A2A mount point
                  (e.g. ``http://localhost:8337/a2a``).

    Returns:
        Populated ``AgentCard`` instance.
    """
    description: str = (
        persona.config.get("description")
        or f"OVOS Persona: {persona.name}"
    )
    # The JSON-RPC route is registered with ``rpc_url="/"`` inside a
    # Starlette app that is mounted at ``base_url``'s path (e.g. "/a2a"), so
    # the actual endpoint has a trailing slash ("/a2a/"). A client that
    # POSTs to the advertised URL without one gets Starlette's 307 redirect,
    # and the a2a-sdk transport does not follow redirects — it would just
    # fail. Advertise the URL clients can actually hit.
    rpc_url = base_url.rstrip("/") + "/"
    return AgentCard(
        name=persona.name,
        description=description,
        version="1.0",
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="chat",
                name="Chat",
                description="Multi-turn conversation with an OVOS persona",
                input_modes=["text"],
                output_modes=["text"],
            )
        ],
        default_input_modes=["text"],
        default_output_modes=["text"],
        supported_interfaces=[
            AgentInterface(url=rpc_url, protocol_binding="JSONRPC")
        ],
    )


class OVOSPersonaAgentExecutor(AgentExecutor):
    """
    A2A ``AgentExecutor`` that delegates to an OVOS ``Persona``.

    ``Persona.stream()`` is synchronous; it is offloaded to a thread via
    ``asyncio.to_thread`` so the event loop is never blocked.

    Sentence chunks are emitted as ``TaskArtifactUpdateEvent`` events (via
    ``TaskUpdater``), enabling real-time SSE streaming for ``message/stream``
    callers while non-streaming ``message/send`` callers receive the same
    events collected into a final ``Task`` by the A2A framework.
    """

    def __init__(self, persona: "Persona") -> None:
        """Initialize the executor with a loaded persona.

        Args:
            persona: Loaded OVOS ``Persona`` instance.
        """
        self._persona = persona

    async def execute(
        self,
        context: "RequestContext",
        event_queue: "EventQueue",
    ) -> None:
        """
        Handle a single A2A turn: call ``Persona.stream`` and emit events.

        Args:
            context: A2A request context carrying the incoming message.
            event_queue: Queue into which response events are enqueued.
        """
        user_text = context.get_user_input()
        messages = [{"role": "user", "content": user_text}]

        # ``context_id`` is the A2A conversation identifier — the id of the
        # "contextual collection of interactions (tasks and messages)", not of a
        # single task (that is ``task_id``). It is the caller identity this
        # protocol carries, so in CHAT_MEMORY=transparent mode it keys memory.
        # Persona.stream is synchronous — run in a thread pool

        # The framework requires a Task object to exist before any
        # TaskStatusUpdateEvent/TaskArtifactUpdateEvent referencing it.
        await event_queue.enqueue_event(
            Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            )
        )

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )
        await updater.start_work()

        try:
            chunks = await asyncio.to_thread(
                lambda: list(run_stream(self._persona, messages,
                                        session_id=context.context_id))
            )
        except PersonaNoAnswerError as exc:
            # No handler produced an answer — terminate the task cleanly with
            # a failed status carrying the reason, rather than letting the
            # exception escape execute() and leave the task stuck non-terminal.
            await updater.failed(
                message=updater.new_agent_message([Part(text=str(exc))])
            )
            return

        # All chunks belong to the same streamed artifact: the first chunk
        # creates it (append=False), subsequent chunks append to it.
        non_empty = [c for c in chunks if c]
        last_index = len(non_empty) - 1
        started = False
        for i, chunk in enumerate(non_empty):
            await updater.add_artifact(
                parts=[Part(text=chunk)],
                artifact_id="response",
                name="response",
                append=started,
                last_chunk=(i == last_index),
            )
            started = True

        await updater.complete()

    async def cancel(
        self,
        context: "RequestContext",
        event_queue: "EventQueue",
    ) -> None:
        """
        Handle a cancellation request.

        Args:
            context: A2A request context.
            event_queue: Queue for the cancellation acknowledgement event.
        """
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )
        await updater.cancel()


def create_a2a_application(
    persona: "Persona",
    base_url: str = "http://localhost:8337/a2a",
) -> "Starlette":
    """
    Build a ``Starlette`` app exposing the A2A agent-card and JSON-RPC
    routes for the given persona.

    The returned app can be mounted directly onto a FastAPI app::

        starlette_a2a = create_a2a_application(persona, base_url)
        fastapi_app.mount("/a2a", starlette_a2a)

    Args:
        persona: Loaded OVOS ``Persona`` instance.
        base_url: Public base URL for the A2A mount point, included in the
                  Agent Card so clients can discover the correct endpoint.

    Returns:
        Configured ``Starlette`` app (mountable ASGI app), exposing
        ``GET /.well-known/agent-card.json`` and ``POST /`` (JSON-RPC).

    Raises:
        RuntimeError: If ``a2a-sdk`` is not installed.
    """
    if not _A2A_AVAILABLE:
        raise RuntimeError(
            "a2a-sdk is not installed. "
            "Install it with: uv pip install 'ovos-persona-server[a2a]'"
        )

    card = _agent_card(persona, base_url)
    executor = OVOSPersonaAgentExecutor(persona)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = (
        # 1.x well-known path (default: /.well-known/agent-card.json)
        list(create_agent_card_routes(agent_card=card))
        # 0.3.x well-known path, so already-deployed 0.3.x clients can still
        # discover this agent.
        + list(create_agent_card_routes(agent_card=card, card_url="/.well-known/agent.json"))
        # enable_v0_3_compat also accepts 0.3.x method names (message/send,
        # message/stream) on the same JSON-RPC endpoint, instead of
        # rejecting them with -32009.
        + list(
            create_jsonrpc_routes(
                request_handler=handler, rpc_url="/", enable_v0_3_compat=True
            )
        )
    )
    app = Starlette(routes=routes)
    app.agent_card = card
    return app
