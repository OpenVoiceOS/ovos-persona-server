# Licensed under the Apache License, Version 2.0
"""Google Gemini-compatible API endpoints."""
import json
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from ovos_persona import Persona

from ovos_persona_server.embeddings import embed_texts, get_embeddings_backend
from ovos_persona_server.persona import (
    get_default_persona, run_chat, run_stream, resolve_persona, PersonaNoAnswerError,
)
from ovos_persona_server.schemas.gemini import (
    GeminiBatchEmbedContentsRequest,
    GeminiBatchEmbedContentsResponse,
    GeminiCandidate,
    GeminiContent,
    GeminiContentEmbedding,
    GeminiEmbedContentRequest,
    GeminiEmbedContentResponse,
    GeminiPart,
    GeminiRequest,
    GeminiResponse,
)

gemini_router = APIRouter(prefix="/gemini/v1beta/models", tags=["gemini"])


def _normalise_messages(request: GeminiRequest) -> List[Dict[str, str]]:
    """Normalise Gemini contents to persona-compatible message list.

    Maps Gemini 'model' role to 'assistant'. Prepends system instruction if present.

    Args:
        request: Gemini API request.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    messages: List[Dict[str, str]] = []
    if request.systemInstruction:
        sys_text = " ".join(p.text for p in request.systemInstruction.parts)
        messages.append({"role": "system", "content": sys_text})
    for content in request.contents:
        role = "assistant" if content.role == "model" else content.role
        text = " ".join(p.text for p in content.parts)
        messages.append({"role": role, "content": text})
    return messages


def _build_response(text: str, model_id: str) -> GeminiResponse:
    """Build a GeminiResponse wrapping the given text.

    Args:
        text: Generated response text.
        model_id: Model identifier for the response.

    Returns:
        GeminiResponse with one candidate.
    """
    return GeminiResponse(
        candidates=[
            GeminiCandidate(
                content=GeminiContent(role="model", parts=[GeminiPart(text=text)]),
                finishReason="STOP",
                index=0,
            )
        ]
    )


@gemini_router.post("/{model_id}:generateContent", response_model=None)
async def generate_content(
        model_id: str,
        request: GeminiRequest,
        key: Optional[str] = Query(default=None),
        persona: Persona = Depends(get_default_persona),
) -> JSONResponse:
    """Generate content (Google Gemini-compatible).

    Args:
        model_id: Gemini model identifier (e.g. gemini-pro), used in path only.
        request: Gemini request body with contents and optional system instruction.
        key: API key query param (accepted, ignored).
        persona: Injected persona instance.

    Returns:
        GeminiResponse JSON with generated content.

    Raises:
        HTTPException: 500 if persona chat fails.
    """
    persona = resolve_persona(model_id, persona)
    messages = _normalise_messages(request)
    try:
        text = run_chat(persona, messages)
    except PersonaNoAnswerError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Persona chat failed: {exc}") from exc
    return JSONResponse(_build_response(text, model_id).model_dump())


@gemini_router.post("/{model_id}:streamGenerateContent", response_model=None)
async def stream_generate_content(
        model_id: str,
        request: GeminiRequest,
        key: Optional[str] = Query(default=None),
        persona: Persona = Depends(get_default_persona),
) -> StreamingResponse:
    """Stream generated content (Google Gemini-compatible).

    Args:
        model_id: Gemini model identifier (e.g. gemini-pro), used in path only.
        request: Gemini request body with contents and optional system instruction.
        key: API key query param (accepted, ignored).
        persona: Injected persona instance.

    Returns:
        SSE stream of GeminiResponse JSON objects.
    """
    persona = resolve_persona(model_id, persona)
    messages = _normalise_messages(request)

    async def _stream() -> AsyncGenerator[str, None]:
        """Yield SSE events with GeminiResponse chunks."""
        try:
            for chunk in run_stream(persona, messages):
                if chunk:
                    resp = _build_response(chunk, model_id)
                    yield f"data: {json.dumps(resp.model_dump())}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@gemini_router.post("/{model_id}:embedContent", response_model=None)
async def embed_content(
        model_id: str,
        request: GeminiEmbedContentRequest,
        key: Optional[str] = Query(default=None),
        embedder=Depends(get_embeddings_backend),
) -> JSONResponse:
    """Embed a single content (Google Gemini-compatible).

    Delegates to the shared embeddings backend (:func:`get_embeddings_backend`),
    the same service used by every other vendor embedding surface.

    Args:
        model_id: Gemini embedding model identifier, used in path only.
        request: Gemini embedContent request body.
        key: API key query param (accepted, ignored).
        embedder: Injected shared embeddings backend.

    Returns:
        GeminiEmbedContentResponse JSON with one embedding.

    Raises:
        HTTPException: 501 if no embeddings backend is available; 500 on backend failure.
    """
    text = " ".join(p.text for p in request.content.parts)
    vector = embed_texts(embedder, [text])[0]
    response = GeminiEmbedContentResponse(embedding=GeminiContentEmbedding(values=vector))
    return JSONResponse(response.model_dump())


@gemini_router.post("/{model_id}:batchEmbedContents", response_model=None)
async def batch_embed_contents(
        model_id: str,
        request: GeminiBatchEmbedContentsRequest,
        key: Optional[str] = Query(default=None),
        embedder=Depends(get_embeddings_backend),
) -> JSONResponse:
    """Embed a batch of contents (Google Gemini-compatible).

    Args:
        model_id: Gemini embedding model identifier, used in path only.
        request: Gemini batchEmbedContents request body.
        key: API key query param (accepted, ignored).
        embedder: Injected shared embeddings backend.

    Returns:
        GeminiBatchEmbedContentsResponse JSON with one embedding per request.

    Raises:
        HTTPException: 501 if no embeddings backend is available; 500 on backend failure.
    """
    texts = [" ".join(p.text for p in item.content.parts) for item in request.requests]
    vectors = embed_texts(embedder, texts)
    response = GeminiBatchEmbedContentsResponse(
        embeddings=[GeminiContentEmbedding(values=v) for v in vectors]
    )
    return JSONResponse(response.model_dump())
