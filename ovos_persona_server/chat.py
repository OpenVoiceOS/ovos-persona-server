"""
FastAPI router for OpenAI-compatible chat and completions endpoints.

This module defines the API endpoints for chat completions (modern)
and legacy text completions, allowing interaction with the OVOS Persona
system using OpenAI's API specifications.
"""

import json
import random
import string
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict, Any, Union

from fastapi import APIRouter, HTTPException, status, Depends, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from ovos_persona import Persona

from ovos_persona_server.persona import get_default_persona
from ovos_persona_server.schemas.openai_chat import (
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
    ChatCompletionResponseMessage, ChatCompletionChoice, ChatCompletionStreamChoice,
    CompletionUsage, FinishReason,
    CreateCompletionRequest, CreateCompletionResponse
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages the lifespan of the FastAPI application, ensuring the default persona is loaded.
    """
    await get_default_persona()
    yield
    # No specific shutdown logic needed for these dependencies currently


chat_router = APIRouter(prefix="/v1", tags=["openai"], lifespan=lifespan)


@chat_router.post(
    "/chat/completions",
    response_model=Union[CreateChatCompletionResponse, CreateChatCompletionStreamResponse],
    status_code=status.HTTP_200_OK
)
async def chat_completions(
        request_body: CreateChatCompletionRequest,
        persona: Persona = Depends(get_default_persona)
) -> Union[JSONResponse, StreamingResponse]:
    """
    Handles OpenAI-compatible chat completions, supporting both non-streaming and streaming.

    NOTE: only 'messages' and 'stream' are currently handled, every other parameter is ignored

    Args:
        request_body (CreateChatCompletionRequest): The request body containing messages and
                                              other chat completion parameters.
        persona (Persona): The persona instance, injected by FastAPI's dependency.

    Returns:
        Union[JSONResponse, StreamingResponse]: A JSON response for non-streaming
                                                or a StreamingResponse for streaming requests.

    Raises:
        HTTPException: If the persona chat fails.
    """
    stream: bool = request_body.stream
    # Convert Pydantic message models to dicts for persona.chat/stream
    messages: List[Dict[str, Any]] = [msg.model_dump(exclude_unset=True) for msg in request_body.messages]

    completion_id: str = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
    completion_timestamp: int = int(time.time())
    if not stream:
        try:
            # Call persona's chat method
            content: str = persona.chat(messages)

            # Basic token count estimation
            prompt_tokens: int = sum(len(msg.get("content", "").split()) for msg in messages) if messages else 0
            completion_tokens: int = len(content.split())

            # Construct the response
            response_message: ChatCompletionResponseMessage = ChatCompletionResponseMessage(
                role="assistant", content=content
            )
            choice: ChatCompletionChoice = ChatCompletionChoice(
                index=0,
                message=response_message,
                finish_reason=FinishReason.STOP  # Assuming 'stop' for non-streaming
            )
            usage: CompletionUsage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )

            # Return the full CreateChatCompletionResponse model
            return JSONResponse(content=CreateChatCompletionResponse(
                id=f"chatcmpl-{completion_id}",
                object="chat.completion",
                created=completion_timestamp,
                model=persona.name,  # Use persona's name as the model
                choices=[choice],
                usage=usage,
                # Add other fields from request_body if needed for response, e.g., tool_choice, seed
                tool_choice=request_body.tool_choice,
                seed=request_body.seed,
                temperature=request_body.temperature,
                top_p=request_body.top_p,
                presence_penalty=request_body.presence_penalty,
                frequency_penalty=request_body.frequency_penalty,
                input_user=request_body.user,  # Map user to input_user
                tools=request_body.tools,
                metadata=request_body.response_metadata,  # Map response_metadata to metadata
                response_format=request_body.response_format,
                parallel_tool_calls=request_body.parallel_tool_calls
            ).model_dump(exclude_unset=True))

        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Persona chat failed: {e}") from e

    async def streaming_chat_response() -> AsyncGenerator[str, None]:
        """
        Asynchronously streams chat completion chunks.
        """
        # Initial chunk with role
        initial_chunk = CreateChatCompletionStreamResponse(
            id=f"chatcmpl-{completion_id}",
            object="chat.completion.chunk",
            created=completion_timestamp,
            model=persona.name,
            choices=[
                ChatCompletionStreamChoice(index=0, delta={'role': 'assistant', "content": ""}, finish_reason=None)
            ],
            system_fingerprint=None,
            usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        ).model_dump_json(exclude_unset=True)
        yield f"data: {initial_chunk}\n\n"

        current_completion_tokens: int = 0
        try:
            for chunk in persona.stream(messages):
                if chunk:  # Only send if chunk is not empty
                    current_completion_tokens += len(chunk.split())  # Basic token count
                    stream_chunk = CreateChatCompletionStreamResponse(
                        id=f"chatcmpl-{completion_id}",
                        object="chat.completion.chunk",
                        created=completion_timestamp,
                        model=persona.name,
                        choices=[
                            ChatCompletionStreamChoice(index=0, delta={'role': 'assistant', 'content': chunk}, finish_reason=None)
                        ],
                        system_fingerprint=None,
                        usage=CompletionUsage(prompt_tokens=0, completion_tokens=current_completion_tokens,
                                              total_tokens=current_completion_tokens)
                    ).model_dump_json(exclude_unset=True)
                    yield f"data: {stream_chunk}\n\n"
        except Exception as e:
            # Send error as part of stream
            error_chunk = json.dumps({'error': str(e), 'done': True})
            yield f"data: {error_chunk}\n\n"
            return

        # Final chunk with finish reason and usage if requested
        final_chunk_data: Dict[str, Any] = {
            'id': f"chatcmpl-{completion_id}",
            'object': "chat.completion.chunk",
            'created': completion_timestamp,
            'model': persona.name,
            'choices': [{
                'index': 0,
                'delta': {},
                'finish_reason': FinishReason.STOP.value  # Assuming 'stop' for streaming completion
            }]
        }

        # Include usage if stream_options.include_usage is true
        if request_body.stream_options and request_body.stream_options.include_usage:
            prompt_tokens = sum(len(msg.get("content", "").split()) for msg in messages) if messages else 0
            usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=current_completion_tokens,
                total_tokens=prompt_tokens + current_completion_tokens
            )
            final_chunk_data['usage'] = usage.model_dump(exclude_unset=True)

        final_chunk = json.dumps(final_chunk_data)
        yield f"data: {final_chunk}\n\n"
        yield "data: [DONE]\n\n"  # Signal end of stream

    return StreamingResponse(streaming_chat_response(), media_type="text/event-stream")


@chat_router.post("/completions", response_model=CreateCompletionResponse, status_code=status.HTTP_200_OK)
async def create_completion(
        request_body: CreateCompletionRequest,
        persona: Persona = Depends(get_default_persona)
) -> Union[JSONResponse, StreamingResponse]:
    """
    Handles legacy OpenAI completions API requests.

    NOTE: only 'prompt' and 'stream' are currently handled, every other parameter is ignored

    Args:
        request_body (CreateCompletionRequest): The request body containing the prompt
                                                and other completion parameters.
        persona (Persona): The persona instance, injected by FastAPI's dependency.

    Returns:
        Union[JSONResponse, StreamingResponse]: A JSON response for non-streaming
                                                or a StreamingResponse for streaming requests.

    Raises:
        HTTPException: If the prompt format is invalid or persona completion fails.
    """
    stream: bool = request_body.stream
    prompt: Union[str, List[str], List[int], List[List[int]]] = request_body.prompt

    # Convert prompt to a list of messages for persona.chat/stream
    # The persona expects a list of dicts with 'role' and 'content'
    messages: List[Dict[str, str]]
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list) and all(isinstance(p, str) for p in prompt):
        # If it's a list of strings, join them into a single message.
        messages = [{"role": "user", "content": "\n".join(prompt)}]
    elif isinstance(prompt, list) and all(isinstance(p, int) for p in prompt):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to generate completion: token_ids support not yet implemented")
    elif isinstance(prompt, list) and all(isinstance(p, list) and all(isinstance(i, int) for i in p) for p in prompt):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to generate completion: token_ids support not yet implemented")
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid prompt format.")

    completion_id: str = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
    completion_timestamp: int = int(time.time())

    if not stream:
        try:
            content: str = persona.chat(messages)

            prompt_tokens: int = sum(len(msg.get("content", "").split()) for msg in messages) if messages else 0
            completion_tokens: int = len(content.split())

            # For legacy completions, choices is a list of dicts.
            response_choice_data = {
                "text": content,
                "index": 0,
                "logprobs": None,  # Not supported in this basic implementation
                "finish_reason": FinishReason.STOP.value  # Assuming 'stop' for non-streaming
            }
            usage: CompletionUsage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )

            return JSONResponse(content=CreateCompletionResponse(
                id=f"cmpl-{completion_id}",
                object="text_completion",
                created=completion_timestamp,
                model=request_body.model,  # Use the model specified in the request
                choices=[response_choice_data],
                usage=usage,
                system_fingerprint=None  # Not provided by persona
            ).model_dump(exclude_unset=True))

        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Persona completion failed: {e}") from e

    async def streaming_completion_response() -> AsyncGenerator[str, None]:
        """
        Asynchronously streams legacy completion chunks.
        """
        current_completion_tokens: int = 0
        try:
            for chunk in persona.stream(messages):
                if chunk:
                    current_completion_tokens += len(chunk.split())
                    # Legacy completion stream format
                    yield f"data: {json.dumps({
                        'id': f"cmpl-{completion_id}",
                        'object': "text_completion",
                        'created': completion_timestamp,
                        'model': request_body.model,
                        'choices': [{
                            'text': chunk,
                            'index': 0,
                            'logprobs': None,  # Not supported in this basic implementation
                            'finish_reason': None
                        }]
                    })}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
            return

        # Final chunk with finish reason
        yield f"data: {json.dumps({
            'id': f"cmpl-{completion_id}",
            'object': "text_completion",
            'created': completion_timestamp,
            'model': request_body.model,
            'choices': [{
                'text': "",  # Empty text for the final chunk
                'index': 0,
                'logprobs': None,
                'finish_reason': FinishReason.STOP.value
            }]
        })}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(streaming_completion_response(), media_type="text/event-stream")
