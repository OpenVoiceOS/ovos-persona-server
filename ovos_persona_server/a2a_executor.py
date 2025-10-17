from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps.jsonrpc import A2AFastAPIApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError
from fastapi import FastAPI
from ovos_persona import Persona


class PersonaAgentExecutor(AgentExecutor):

    def __init__(self, persona: Persona):
        self.persona = persona

    async def execute(
            self,
            context: RequestContext,
            event_queue: EventQueue,
    ) -> None:
        query: str = context.get_user_input()
        # TODO - placeholder
        result = self.persona.chat([{"role": "user", "content": query}])
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(
            self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())


def get_a2a_app(persona: Persona, version: str, url: str) -> FastAPI:
    skill = AgentSkill(
        id='ovos_persona',
        name='chat',
        description=f'chat with {persona.name}',
        tags=['chat'],
        examples=['hello', 'explain quantum mechanics in simple terms'],
    )

    public_agent_card = AgentCard(
        name=persona.name,
        description=f"chat with {persona.name}",
        url=url,
        version=version or '0.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True), # TODO - ensure streaming supported
        skills=[skill],
        supports_authenticated_extended_card=False
    )

    request_handler = DefaultRequestHandler(
        agent_executor=PersonaAgentExecutor(persona),
        task_store=InMemoryTaskStore(),
    )

    server = A2AFastAPIApplication(
        agent_card=public_agent_card,
        http_handler=request_handler
    )

    return server.build()
