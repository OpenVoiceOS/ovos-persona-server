"""
FastAPI router for OpenAI-compatible chat and completions endpoints.

This module defines the API endpoints for chat completions (modern)
and legacy text completions, allowing interaction with the OVOS Persona
system using OpenAI's API specifications.
"""

import base64
import json
import random
import string
import struct
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict, Any, Literal, Optional, Union

from fastapi import APIRouter, HTTPException, status, Depends, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from ovos_persona import Persona
from pydantic import BaseModel, Field

from ovos_persona_server.embeddings import get_embeddings_backend, embed_texts, backend_model_name
from ovos_persona_server.persona import get_default_persona
from ovos_persona_server.schemas.openai_chat import (
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
    ChatCompletionResponseMessage, ChatCompletionChoice, ChatCompletionStreamChoice,
    CompletionUsage, FinishReason,
    CreateCompletionRequest, CreateCompletionResponse
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Ensure the default persona is ready before serving requests."""
    await get_default_persona()
    yield


chat_router = APIRouter(prefix="/openai/v1", tags=["openai"], lifespan=lifespan)


@chat_router.post(
    "/chat/completions",
    response_model=None,
    status_code=status.HTTP_200_OK
)
async def chat_completions(
        request_body: CreateChatCompletionRequest,
        persona: Persona = Depends(get_default_persona)
) -> Union[JSONResponse, StreamingResponse]:
    """Handle OpenAI-compatible chat completions (streaming and non-streaming).

    Only ``messages`` and ``stream`` are forwarded to the persona; all other
    OpenAI parameters are accepted and silently ignored.

    Args:
        request_body: Chat completion request containing messages and options.
        persona: Injected persona instance.

    Returns:
        JSON response (non-streaming) or SSE StreamingResponse.

    Raises:
        HTTPException: If the persona chat call raises an unexpected exception.
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
        """Yield SSE data events in OpenAI streaming format."""
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


@chat_router.post("/completions", response_model=None, status_code=status.HTTP_200_OK)
async def create_completion(
        request_body: CreateCompletionRequest,
        persona: Persona = Depends(get_default_persona)
) -> Union[JSONResponse, StreamingResponse]:
    """Handle legacy OpenAI text-completion API requests.

    Only ``prompt`` and ``stream`` are forwarded to the persona; other
    parameters are accepted and silently ignored.  Token-array prompts are
    not supported and return 500.

    Args:
        request_body: Completion request with prompt and options.
        persona: Injected persona instance.

    Returns:
        JSON response (non-streaming) or SSE StreamingResponse.

    Raises:
        HTTPException: On unsupported prompt format or persona failure.
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
        """Yield SSE data events in legacy OpenAI text-completion format."""
        current_completion_tokens: int = 0
        try:
            for chunk in persona.stream(messages):
                if chunk:
                    current_completion_tokens += len(chunk.split())
                    # Legacy completion stream format
                    chunk_data = {
                        'id': f"cmpl-{completion_id}",
                        'object': "text_completion",
                        'created': completion_timestamp,
                        'model': request_body.model,
                        'choices': [{
                            'text': chunk,
                            'index': 0,
                            'logprobs': None,
                            'finish_reason': None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
            return

        # Final chunk with finish reason
        final_completion_data = {
            'id': f"cmpl-{completion_id}",
            'object': "text_completion",
            'created': completion_timestamp,
            'model': request_body.model,
            'choices': [{
                'text': "",
                'index': 0,
                'logprobs': None,
                'finish_reason': FinishReason.STOP.value
            }]
        }
        yield f"data: {json.dumps(final_completion_data)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(streaming_completion_response(), media_type="text/event-stream")


@chat_router.get("/models")
async def list_models(persona: Persona = Depends(get_default_persona)) -> JSONResponse:
    """List available models (OpenAI-compatible).

    Args:
        persona: Injected persona instance.

    Returns:
        OpenAI-format models list containing the loaded persona.
    """
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": persona.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ovos",
            }
        ],
    })


class OpenAIEmbeddingsRequest(BaseModel):
    """Request body for POST /v1/embeddings."""

    model: str = Field(default="text-embedding-ada-002", min_length=1)
    input: Union[str, List[str]] = Field(..., description="Text or list of texts to embed")
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = Field(default=None, gt=0)
    user: Optional[str] = None


@chat_router.post("/embeddings")
async def embeddings(
        request_body: OpenAIEmbeddingsRequest,
        embedder=Depends(get_embeddings_backend),
) -> JSONResponse:
    """Generate embeddings (OpenAI-compatible).

    Delegates to the shared, swappable embeddings backend
    (:func:`get_embeddings_backend`) — the same service used by the Ollama
    endpoint and vector-store search. Configure it via ``TEXT_EMBEDDINGS_PLUGIN``
    / ``EMBEDDINGS_URL`` / ``EMBEDDINGS_MODEL`` to point at any embeddings provider.

    Args:
        request_body: OpenAI embeddings request with model and input.
        embedder: Injected shared embeddings backend.

    Returns:
        OpenAI-format embeddings response.

    Raises:
        HTTPException: 501 if no embeddings backend is available; 500 on backend failure.
    """
    texts = request_body.input if isinstance(request_body.input, list) else [request_body.input]
    vectors = embed_texts(embedder, texts)
    # The official openai SDK requests encoding_format="base64" by default and
    # decodes float32 buffers client-side; honour it for SDK compatibility.
    if request_body.encoding_format == "base64":
        encoded = [base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode("ascii")
                   for vec in vectors]
        data = [{"object": "embedding", "embedding": e, "index": i} for i, e in enumerate(encoded)]
    else:
        data = [{"object": "embedding", "embedding": vec, "index": i}
                for i, vec in enumerate(vectors)]
    prompt_tokens = sum(len(t.split()) for t in texts)

    return JSONResponse({
        "object": "list",
        "data": data,
        "model": backend_model_name(embedder, request_body.model),
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    })
