"""
A2A (Agent-to-Agent) server adapter for OVOS Persona Server.

Exposes the loaded persona as an A2A-compatible agent server, allowing any
A2A client to interact with OVOS personas using the standard A2A protocol.

Endpoints (mounted at the path passed to ``create_a2a_application``):
  GET  /.well-known/agent.json   — Agent Card (capabilities, skills, URL)
  POST /                         — JSON-RPC 2.0: message/send, message/stream

A2A spec: https://google.github.io/A2A/
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from ovos_persona_server.persona import run_stream

LOG = logging.getLogger(__name__)

# Sentinel names so monkeypatch works without a2a-sdk installed
AgentCard = None
AgentCapabilities = None
AgentSkill = None
Artifact = None
Part = None
TextPart = None
TaskArtifactUpdateEvent = None
TaskStatusUpdateEvent = None
TaskState = None
TaskStatus = None
AgentExecutor = object  # base class fallback
EventQueue = None
RequestContext = None
DefaultRequestHandler = None
InMemoryTaskStore = None
A2AStarletteApplication = None

try:
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.events import EventQueue
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import (
        AgentCard,
        AgentCapabilities,
        AgentSkill,
        Artifact,
        Part,
        TaskArtifactUpdateEvent,
        TaskState,
        TaskStatus,
        TaskStatusUpdateEvent,
        TextPart,
    )

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
    return AgentCard(
        name=persona.name,
        description=description,
        url=base_url,
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
    )


class OVOSPersonaAgentExecutor(AgentExecutor):
    """
    A2A ``AgentExecutor`` that delegates to an OVOS ``Persona``.

    ``Persona.stream()`` is synchronous; it is offloaded to a thread via
    ``asyncio.to_thread`` so the event loop is never blocked.

    Sentence chunks are emitted as ``TaskArtifactUpdateEvent`` events,
    enabling real-time SSE streaming for ``message/stream`` callers while
    non-streaming ``message/send`` callers receive the same events collected
    into a final ``Task`` by the A2A framework.
    """

    def __init__(self, persona: "Persona") -> None:
        """Initialize the executor with a loaded persona.

        Args:
            persona: Loaded OVOS ``Persona`` instance.
        """
        self._persona = persona

    @staticmethod
    def _extract_user_text(message: object) -> str:
        """
        Extract plain text from an incoming A2A ``Message``.

        Args:
            message: A2A ``Message`` object from ``RequestContext``.

        Returns:
            Concatenated text from all ``TextPart`` parts.
        """
        parts = getattr(message, "parts", None) or []
        texts = []
        for part in parts:
            root = getattr(part, "root", None)
            if isinstance(root, TextPart):
                texts.append(root.text)
        return " ".join(texts)

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
        user_text = self._extract_user_text(context.message)
        messages = [{"role": "user", "content": user_text}]

        # Persona.stream is synchronous — run in a thread pool
        chunks = await asyncio.to_thread(
            lambda: list(run_stream(self._persona, messages))
        )

        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            artifact = Artifact(
                parts=[Part(root=TextPart(text=chunk))],
                artifact_id=str(i),
                name=f"chunk-{i}",
            )
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    artifact=artifact,
                    append=(i > 0),
                    last_chunk=(i == len(chunks) - 1),
                )
            )

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.completed),
                final=True,
            )
        )

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
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.canceled),
                final=True,
            )
        )


def create_a2a_application(
    persona: "Persona",
    base_url: str = "http://localhost:8337/a2a",
) -> "A2AStarletteApplication":
    """
    Build an ``A2AStarletteApplication`` wrapping the given persona.

    The returned Starlette app can be mounted directly onto a FastAPI app::

        starlette_a2a = create_a2a_application(persona, base_url).build()
        fastapi_app.mount("/a2a", starlette_a2a)

    Args:
        persona: Loaded OVOS ``Persona`` instance.
        base_url: Public base URL for the A2A mount point, included in the
                  Agent Card so clients can discover the correct endpoint.

    Returns:
        Configured ``A2AStarletteApplication`` (call ``.build()`` to get the
        ASGI app for mounting).

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
    )
    return A2AStarletteApplication(agent_card=card, http_handler=handler)
