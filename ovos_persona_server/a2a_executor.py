import asyncio
from typing import Callable, AsyncGenerator

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps.jsonrpc import A2AFastAPIApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    Part,
    TaskState,
    TextPart,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    UnsupportedOperationError,
)
from a2a.utils import new_task, new_agent_text_message
from a2a.utils.errors import ServerError
from fastapi import FastAPI
from ovos_bus_client.session import Session
from ovos_persona import Persona
from ovos_utils.log import LOG

from ovos_persona_server.persona import get_default_persona


class PersonaAgentExecutor(AgentExecutor):

    def __init__(self, persona: Persona):
        self.persona = persona

    async def execute(
            self,
            context: RequestContext,
            event_queue: EventQueue,
    ) -> None:
        query: str = context.get_user_input()

        # A simple non-streaming implementation would be just
        # result = self.persona.chat([{"role": "user", "content": query}])
        # await event_queue.enqueue_event(new_agent_text_message(result))
        # return

        # check if message is continuation of a previous task
        # NOTE: a task in OVOS context can be thought of as an individual conversation
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)

        # streaming response
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        sess = Session(task.id)  # TODO - track memory per task.id
        # in practice all Session is doing for now
        # is getting lang and system_unit from mycroft.conf
        messages = [{"role": "user", "content": query}]

        try:
            response = []
            for sentence in self.persona.stream(messages,
                                                lang=sess.lang,
                                                units=sess.system_unit):
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        sentence,
                        task.context_id,
                        task.id,
                    ),
                )
                response.append(sentence)

            await updater.add_artifact(
                [Part(root=TextPart(text="\n".join(response)))],
                name='full_response',
            )
            await updater.complete()  # marks task as complete and publishes a final update

        except Exception as e:
            LOG.error(f'An error occurred while streaming the response: {e}')
            raise ServerError(error=InternalError()) from e

    async def cancel(
            self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())


def get_a2a_app(version: str,
                domain: str,
                lifespan: Callable[[FastAPI], AsyncGenerator[None, None]],
                rpc_url="/a2a",
                title="OpenVoiceOS Persona Server",
                description="OpenAI/Ollama compatible API for OVOS Personas and Solvers") -> FastAPI:

    persona: Persona = asyncio.run(get_default_persona())

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
        url=domain + rpc_url,
        version=version or '0.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
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

    return server.build(title=title,
                        rpc_url=f"{rpc_url}",
                        description=description,
                        lifespan=lifespan)
