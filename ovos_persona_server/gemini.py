# Licensed under the Apache License, Version 2.0
"""Google Gemini-compatible API endpoints."""
import json
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from ovos_persona import Persona

from ovos_persona_server.persona import get_default_persona
from ovos_persona_server.schemas.gemini import (
    GeminiCandidate,
    GeminiContent,
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
    messages = _normalise_messages(request)
    try:
        text = persona.chat(messages)
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
    messages = _normalise_messages(request)

    async def _stream() -> AsyncGenerator[str, None]:
        """Yield SSE events with GeminiResponse chunks."""
        try:
            for chunk in persona.stream(messages):
                if chunk:
                    resp = _build_response(chunk, model_id)
                    yield f"data: {json.dumps(resp.model_dump())}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
