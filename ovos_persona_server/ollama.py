"""
FastAPI router for Ollama-compatible API endpoints.

This module provides endpoints for chat, text generation, and listing
available models, adhering to the Ollama API specification.
"""

import datetime
import json
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Union, AsyncGenerator, Optional

from fastapi import Depends, status, APIRouter, HTTPException
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from ovos_bus_client.session import SessionManager
from ovos_persona import Persona
from pydantic import BaseModel, Field


from ovos_persona_server.embeddings import get_embeddings_backend, embed_texts
from ovos_persona_server.persona import get_default_persona
from ovos_persona_server.schemas.ollama import (
    OllamaChatResponse,
    OllamaTagsResponse,
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaModelDetails,
    OllamaModel,
    OllamaChatMessage,
    OllamaEmbedRequest,
    OllamaEmbedResponse,
    OllamaEmbeddingsRequest,
    OllamaEmbeddingsResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Ensure the default persona is ready before serving requests."""
    await get_default_persona()
    yield


ollama_router = APIRouter(prefix="/ollama/api", tags=["ollama"], lifespan=lifespan)


def timestamp() -> str:
    """
    Returns an ISO 8601 formatted UTC timestamp with microseconds.

    Returns:
        str: The current UTC timestamp in ISO 8601 format.
    """
    return datetime.datetime.utcnow().isoformat(timespec='microseconds') + "Z"


@ollama_router.post("/chat", response_model=OllamaChatResponse, status_code=status.HTTP_200_OK)
async def chat_ollama(request_body: OllamaChatRequest, persona: Persona = Depends(get_default_persona)) -> Union[
    JSONResponse, StreamingResponse]:
    """Handle Ollama-compatible chat requests.

    Only ``messages`` and ``stream`` are forwarded to the persona.  Tool calls,
    image attachments, and generation options are accepted but not yet processed.

    Args:
        request_body: Ollama chat request.
        persona: Injected persona instance.

    Returns:
        JSON response (non-streaming) or NDJSON StreamingResponse.

    Raises:
        HTTPException: If messages are empty or the persona call fails.
    """
    messages: List[OllamaChatMessage] = request_body.messages
    stream: bool = request_body.stream
    # Other parameters from request_body (tools, think, format, options, keep_alive)
    # are currently not used by the persona's chat/stream methods.

    if not messages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Messages are required.")

    sess = SessionManager().get()
    ts: str = timestamp()

    # Placeholder for usage metrics, replace with actual values if Persona can provide them
    # For now, these are illustrative defaults based on Ollama API spec.
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0
    done_reason: Optional[str] = "stop"  # Default reason for completion

    # Convert OllamaChatMessage objects to a list of dicts for persona.chat/stream
    persona_messages: List[Dict[str, str]] = [msg.model_dump(exclude_unset=True) for msg in messages]

    if not stream:
        try:
            start_time: float = time.time()
            content: str = persona.chat(persona_messages, lang=sess.lang, units=sess.system_unit)
            end_time: float = time.time()
            total_duration = int((end_time - start_time) * 1_000_000_000)  # Nanoseconds

            return JSONResponse(content=OllamaChatResponse(
                model=persona.name,
                created_at=ts,
                message={"role": "assistant", "content": content},
                done=True,
                total_duration=total_duration,
                load_duration=load_duration,  # Placeholder
                prompt_eval_count=prompt_eval_count,  # Placeholder
                prompt_eval_duration=prompt_eval_duration,  # Placeholder
                eval_count=eval_count,  # Placeholder
                eval_duration=eval_duration,  # Placeholder
                done_reason=done_reason
            ).model_dump(exclude_unset=True))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Persona chat failed: {e}") from e

    async def streaming_ollama_chat_response() -> AsyncGenerator[str, None]:
        """Yield NDJSON lines in Ollama streaming chat format."""
        start_time: float = time.time()
        # Placeholders for metrics in streaming chunks
        streaming_load_duration: int = 0
        streaming_prompt_eval_count: int = 0
        streaming_prompt_eval_duration: int = 0
        streaming_eval_count: int = 0
        streaming_eval_duration: int = 0

        try:
            for chunk in persona.stream(persona_messages, lang=sess.lang, units=sess.system_unit):
                if chunk:
                    # Increment eval_count for each chunk (approximate)
                    streaming_eval_count += len(chunk.split())  # Simple token count approximation
                    yield json.dumps({
                        "model": persona.name,
                        "created_at": ts,
                        "message": {"role": "assistant", "content": chunk},
                        "done": False,
                        # Update total_duration dynamically for streaming chunks
                        "total_duration": int((time.time() - start_time) * 1_000_000_000),
                        "load_duration": streaming_load_duration,
                        "prompt_eval_count": streaming_prompt_eval_count,
                        "prompt_eval_duration": streaming_prompt_eval_duration,
                        "eval_count": streaming_eval_count,
                        "eval_duration": streaming_eval_duration  # This would be cumulative in a real scenario
                    }) + "\n"
        except Exception as e:
            # Handle streaming errors gracefully
            yield json.dumps({"error": str(e), "done": True, "done_reason": "error"}) + "\n"
            return  # Stop the generator

        end_time: float = time.time()
        streaming_total_duration: int = int((end_time - start_time) * 1_000_000_000)

        yield json.dumps({
            "model": persona.name,
            "created_at": ts,
            "message": {"role": "assistant", "content": ""},  # Empty content for final done chunk
            "done": True,
            "total_duration": streaming_total_duration,
            "load_duration": streaming_load_duration,
            "prompt_eval_count": streaming_prompt_eval_count,
            "prompt_eval_duration": streaming_prompt_eval_duration,
            "eval_count": streaming_eval_count,
            "eval_duration": streaming_eval_duration,
            "done_reason": done_reason
        }) + "\n"

    return StreamingResponse(streaming_ollama_chat_response(), media_type="application/json")


@ollama_router.post("/generate", response_model=None, status_code=status.HTTP_200_OK)
async def generate_ollama(request_body: OllamaGenerateRequest, persona: Persona = Depends(get_default_persona)) -> \
        Union[JSONResponse, StreamingResponse]:
    """Handle Ollama-compatible text generation requests.

    ``prompt``, ``system``, and ``stream`` are forwarded to the persona.
    Images, think-tags, format, options, template, raw, keep_alive, and context
    are accepted but not yet processed.

    Args:
        request_body: Ollama generate request.
        persona: Injected persona instance.

    Returns:
        JSON response (non-streaming) or NDJSON StreamingResponse.

    Raises:
        HTTPException: If prompt is empty or the persona call fails.
    """
    prompt: str = request_body.prompt
    stream: bool = request_body.stream
    system: Optional[str] = request_body.system
    suffix: Optional[str] = request_body.suffix
    # Other parameters from request_body (images, think, format, options, template, raw, keep_alive, context)
    # are currently not used by the persona's chat/stream methods.

    if not prompt:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt is required.")

    # Construct messages for persona based on Ollama generate request
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    ts: str = timestamp()

    # Placeholders for usage metrics
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0
    done_reason: Optional[str] = "stop"  # Default reason for completion

    sess = SessionManager().get()

    if not stream:
        try:
            start_time: float = time.time()
            # Use persona.chat for non-streaming generation
            content: str = persona.chat(messages, lang=sess.lang, units=sess.system_unit)
            end_time: float = time.time()
            total_duration = int((end_time - start_time) * 1_000_000_000)

            # Ollama /generate returns the text under "response" (not a chat
            # "message"), matching the streaming path and the official client.
            return JSONResponse(content={
                "model": persona.name,
                "created_at": ts,
                "response": content,
                "done": True,
                "total_duration": total_duration,
                "load_duration": load_duration,
                "prompt_eval_count": prompt_eval_count,
                "prompt_eval_duration": prompt_eval_duration,
                "eval_count": eval_count,
                "eval_duration": eval_duration,
                "done_reason": done_reason,
            })
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Persona generation failed: {e}") from e

    async def streaming_ollama_generate_response() -> AsyncGenerator[str, None]:
        """Yield NDJSON lines in Ollama streaming generate format."""
        start_time: float = time.time()
        # Placeholders for metrics in streaming chunks
        streaming_load_duration: int = 0
        streaming_prompt_eval_count: int = 0
        streaming_prompt_eval_duration: int = 0
        streaming_eval_count: int = 0
        streaming_eval_duration: int = 0

        try:
            # Use persona.stream for streaming generation
            for chunk in persona.stream(messages, lang=sess.lang, units=sess.system_unit):
                if chunk:
                    # Increment eval_count for each chunk (approximate)
                    streaming_eval_count += len(chunk.split())  # Simple token count approximation
                    yield json.dumps({
                        "model": persona.name,
                        "created_at": ts,
                        "response": chunk,  # Ollama /generate uses "response" key for content
                        "done": False,
                        "total_duration": int((time.time() - start_time) * 1_000_000_000),
                        "load_duration": streaming_load_duration,
                        "prompt_eval_count": streaming_prompt_eval_count,
                        "prompt_eval_duration": streaming_prompt_eval_duration,
                        "eval_count": streaming_eval_count,
                        "eval_duration": streaming_eval_duration  # This would be cumulative in a real scenario
                    }) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e), "done": True, "done_reason": "error"}) + "\n"
            return

        end_time: float = time.time()
        streaming_total_duration: int = int((end_time - start_time) * 1_000_000_000)

        yield json.dumps({
            "model": persona.name,
            "created_at": ts,
            "response": "",  # Empty content for final done chunk in /generate
            "done": True,
            "total_duration": streaming_total_duration,
            "load_duration": streaming_load_duration,
            "prompt_eval_count": streaming_prompt_eval_count,
            "prompt_eval_duration": streaming_prompt_eval_duration,
            "eval_count": streaming_eval_count,
            "eval_duration": streaming_eval_duration,
            "done_reason": done_reason
        }) + "\n"

    return StreamingResponse(streaming_ollama_generate_response(), media_type="application/json")


@ollama_router.get("/tags", response_model=OllamaTagsResponse, status_code=status.HTTP_200_OK)
async def tags(persona: Persona = Depends(get_default_persona)) -> OllamaTagsResponse:
    """Return a list of available Ollama-compatible models.

    Exposes the loaded persona as a single model entry; solver modules appear
    as model families.

    Args:
        persona: Injected persona instance.

    Returns:
        Response containing the list of models.
    """

    # Get loaded solver modules from the persona, which can be thought of as "families"
    solvers: List[str] = list(persona.solvers.loaded_modules.keys())
    # Get models configured for each solver, if any
    models_in_config: set[Optional[str]] = {persona.config.get(s, {}).get("model")
                                            for s in persona.solvers.loaded_modules.keys()}
    # Filter out None values and join them for parent_model
    parent_model_str: str = "|".join(filter(None, models_in_config))

    # Construct OllamaModelDetails based on persona information
    details: OllamaModelDetails = OllamaModelDetails(
        format="json",  # JSON definitions are used for persona
        family="ovos-persona",  # Custom family name for this integration
        families=solvers,  # Solvers can be considered as sub-families
        parent_model=parent_model_str,  # Join configured models
        parameter_size="",  # Placeholder, persona doesn't expose this directly
        quantization_level=""  # Placeholder
    )

    # Construct OllamaModel representing the persona
    model: OllamaModel = OllamaModel(
        name=persona.name,  # Use persona's name as model name
        model=persona.name,  # Use persona's name as model identifier
        digest="sha256:placeholder_digest",  # Placeholder digest
        size=0,  # Placeholder size
        modified_at=timestamp(),  # Current timestamp
        details=details
    )
    return OllamaTagsResponse(
        models=[model]
    )


@ollama_router.get("/show")
async def show(persona: Persona = Depends(get_default_persona)) -> JSONResponse:
    """Show model card (Ollama-compatible stub).

    Args:
        persona: Injected persona instance.

    Returns:
        Static model card for the loaded persona.
    """
    solvers: List[str] = list(persona.solvers.loaded_modules.keys())
    models_in_config: set = {persona.config.get(s, {}).get("model")
                             for s in persona.solvers.loaded_modules.keys()}
    parent_model_str: str = "|".join(filter(None, models_in_config))

    details: OllamaModelDetails = OllamaModelDetails(
        format="json",
        family="ovos-persona",
        families=solvers,
        parent_model=parent_model_str,
        parameter_size="",
        quantization_level="",
    )
    model: OllamaModel = OllamaModel(
        name=persona.name,
        model=persona.name,
        digest="sha256:placeholder_digest",
        size=0,
        modified_at=timestamp(),
        details=details,
    )
    return JSONResponse(model.model_dump())


@ollama_router.get("/ps")
async def ps(persona: Persona = Depends(get_default_persona)) -> JSONResponse:
    """List running models (Ollama-compatible stub).

    Args:
        persona: Injected persona instance.

    Returns:
        Ollama /api/ps format with the loaded persona listed as running.
    """
    solvers: List[str] = list(persona.solvers.loaded_modules.keys())
    models_in_config: set = {persona.config.get(s, {}).get("model")
                             for s in persona.solvers.loaded_modules.keys()}
    parent_model_str: str = "|".join(filter(None, models_in_config))

    details: OllamaModelDetails = OllamaModelDetails(
        format="json",
        family="ovos-persona",
        families=solvers,
        parent_model=parent_model_str,
        parameter_size="",
        quantization_level="",
    )
    model: OllamaModel = OllamaModel(
        name=persona.name,
        model=persona.name,
        digest="sha256:placeholder_digest",
        size=0,
        modified_at=timestamp(),
        details=details,
    )
    return JSONResponse({"models": [model.model_dump()]})


class OllamaPullRequest(BaseModel):
    """Request body for POST /api/pull."""

    model: str = Field(..., min_length=1)
    insecure: Optional[bool] = None
    stream: Optional[bool] = None


class OllamaPushRequest(BaseModel):
    """Request body for POST /api/push."""

    model: str = Field(..., min_length=1)
    insecure: Optional[bool] = None
    stream: Optional[bool] = None


@ollama_router.post("/pull")
async def pull(request_body: OllamaPullRequest) -> JSONResponse:
    """Pull a model (Ollama-compatible stub).

    Args:
        request_body: Pull request with model name (accepted, ignored).

    Returns:
        Success status stub.
    """
    return JSONResponse({"status": "success"})


@ollama_router.post("/push")
async def push(request_body: OllamaPushRequest) -> JSONResponse:
    """Push a model (Ollama-compatible stub).

    Args:
        request_body: Push request with model name (accepted, ignored).

    Returns:
        Success status stub.
    """
    return JSONResponse({"status": "success"})


@ollama_router.post("/embed")
async def ollama_embed(
        request_body: OllamaEmbedRequest,
        embedder=Depends(get_embeddings_backend),
) -> JSONResponse:
    """Generate embeddings (Ollama ``/api/embed``, batch).

    Delegates to the shared, swappable embeddings backend
    (:func:`get_embeddings_backend`) — the same service used by the OpenAI
    endpoint and vector-store search. This is the endpoint the official
    ``ollama`` client's ``embed()`` method targets.

    Args:
        request_body: Ollama embed request with model and ``input`` (str or list).
        embedder: Injected shared embeddings backend.

    Returns:
        Ollama-format embeddings response (one vector per input).

    Raises:
        HTTPException: 501 if no embeddings backend is available; 500 on backend failure.
    """
    texts = request_body.input if isinstance(request_body.input, list) else [request_body.input]
    vectors = embed_texts(embedder, texts)
    return JSONResponse(OllamaEmbedResponse(model=request_body.model, embeddings=vectors).model_dump(exclude_none=True))


@ollama_router.post("/embeddings")
async def ollama_embeddings(
        request_body: OllamaEmbeddingsRequest,
        embedder=Depends(get_embeddings_backend),
) -> JSONResponse:
    """Generate a single embedding (legacy Ollama ``/api/embeddings``).

    Delegates to the shared, swappable embeddings backend. This is the endpoint
    the official ``ollama`` client's ``embeddings()`` method targets (single
    ``prompt`` in, single ``embedding`` out).

    Args:
        request_body: Legacy Ollama embeddings request with model and ``prompt``.
        embedder: Injected shared embeddings backend.

    Returns:
        Ollama-format legacy embeddings response (a single vector).

    Raises:
        HTTPException: 501 if no embeddings backend is available; 500 on backend failure.
    """
    vec = embed_texts(embedder, [request_body.prompt])[0]
    return JSONResponse(OllamaEmbeddingsResponse(embedding=vec).model_dump())
