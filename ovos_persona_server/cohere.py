# Licensed under the Apache License, Version 2.0
"""Cohere-compatible API endpoints."""
import json
import random
import string
import time
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from ovos_persona import Persona
from pydantic import BaseModel, Field

from ovos_persona_server.persona import get_default_persona

cohere_router = APIRouter(prefix="/v1", tags=["cohere"])


class CohereChatMessage(BaseModel):
    """A message in the Cohere chat history."""
    role: Literal["USER", "CHATBOT", "SYSTEM"]
    message: str


class CohereChatRequest(BaseModel):
    """Request body for POST /v1/chat."""
    message: str = Field(..., min_length=1)
    model: Optional[str] = Field(default=None, min_length=1)
    chat_history: List[CohereChatMessage] = Field(default_factory=list)
    preamble: Optional[str] = Field(default=None, min_length=1)
    stream: bool = False
    max_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=5.0)


class CohereGenerateRequest(BaseModel):
    """Request body for POST /v1/generate."""
    prompt: str = Field(..., min_length=1)
    model: Optional[str] = Field(default=None, min_length=1)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    stream: bool = False


class CohereEmbedRequest(BaseModel):
    """Request body for POST /v1/embed."""
    texts: List[str] = Field(..., min_length=1)
    model: Optional[str] = Field(default=None, min_length=1)
    input_type: Optional[Literal["search_document", "search_query", "classification", "clustering"]] = None


def _new_id(prefix: str = "") -> str:
    return prefix + "".join(random.choices(string.ascii_letters + string.digits, k=24))


def _role_map(role: str) -> str:
    return {"USER": "user", "CHATBOT": "assistant", "SYSTEM": "system"}.get(role, "user")


@cohere_router.post("/chat", response_model=None)
async def cohere_chat(
        request: CohereChatRequest,
        persona: Persona = Depends(get_default_persona),
        authorization: Optional[str] = Header(default=None),
) -> Union[JSONResponse, StreamingResponse]:
    """Chat (Cohere-compatible).

    Converts chat_history + message to persona message list.

    Args:
        request: Cohere chat request.
        persona: Injected persona instance.
        authorization: Bearer token (accepted, ignored).

    Returns:
        JSON response or newline-delimited SSE stream in Cohere format.
    """
    messages: List[Dict[str, str]] = []
    if request.preamble:
        messages.append({"role": "system", "content": request.preamble})
    for h in request.chat_history:
        messages.append({"role": _role_map(h.role), "content": h.message})
    messages.append({"role": "user", "content": request.message})

    msg_id = _new_id()
    gen_id = _new_id()

    if not request.stream:
        try:
            text = persona.chat(messages)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=str(exc)) from exc
        return JSONResponse({
            "id": msg_id,
            "finish_reason": "COMPLETE",
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
            "text": text,
            "usage": {
                "billed_units": {
                    "input_tokens": sum(len(m["content"].split()) for m in messages),
                    "output_tokens": len(text.split()),
                }
            },
        })

    async def _stream() -> AsyncGenerator[str, None]:
        accumulated = []
        try:
            for chunk in persona.stream(messages):
                if chunk:
                    accumulated.append(chunk)
                    yield json.dumps({"event_type": "text-generation", "text": chunk}) + "\n"
        except Exception as exc:
            yield json.dumps({"event_type": "stream-end", "finish_reason": "ERROR",
                               "error": str(exc)}) + "\n"
            return
        full_text = "".join(accumulated)
        yield json.dumps({
            "event_type": "stream-end",
            "finish_reason": "COMPLETE",
            "response": {
                "id": msg_id,
                "text": full_text,
                "message": {"role": "assistant", "content": [{"type": "text", "text": full_text}]},
            },
        }) + "\n"

    return StreamingResponse(_stream(), media_type="application/json")


@cohere_router.post("/generate", response_model=None)
async def cohere_generate(
        request: CohereGenerateRequest,
        persona: Persona = Depends(get_default_persona),
        authorization: Optional[str] = Header(default=None),
) -> Union[JSONResponse, StreamingResponse]:
    """Generate text (Cohere-compatible).

    Args:
        request: Cohere generate request.
        persona: Injected persona instance.
        authorization: Bearer token (accepted, ignored).

    Returns:
        JSON response or newline-delimited stream in Cohere format.
    """
    messages = [{"role": "user", "content": request.prompt}]
    gen_id = _new_id()
    resp_id = _new_id()

    if not request.stream:
        try:
            text = persona.chat(messages)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=str(exc)) from exc
        return JSONResponse({
            "id": resp_id,
            "generations": [{"id": gen_id, "text": text, "finish_reason": "COMPLETE"}],
        })

    async def _stream() -> AsyncGenerator[str, None]:
        accumulated = []
        try:
            for chunk in persona.stream(messages):
                if chunk:
                    accumulated.append(chunk)
                    yield json.dumps({"text": chunk, "is_finished": False}) + "\n"
        except Exception as exc:
            yield json.dumps({"text": "", "is_finished": True, "finish_reason": "ERROR",
                               "error": str(exc)}) + "\n"
            return
        full_text = "".join(accumulated)
        yield json.dumps({
            "text": "",
            "is_finished": True,
            "finish_reason": "COMPLETE",
            "response": {"id": resp_id, "generations": [{"id": gen_id, "text": full_text}]},
        }) + "\n"

    return StreamingResponse(_stream(), media_type="application/json")


@cohere_router.post("/embed", response_model=None)
async def cohere_embed(
        request: CohereEmbedRequest,
        persona: Persona = Depends(get_default_persona),
        authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    """Generate embeddings (Cohere-compatible).

    Args:
        request: Cohere embed request.
        persona: Injected persona instance.
        authorization: Bearer token (accepted, ignored).

    Returns:
        Embeddings response or 501 if no solver.

    Raises:
        HTTPException: 501 if no embeddings solver configured.
    """
    embed_solver = None
    for solver in persona.solvers.loaded_modules.values():
        if hasattr(solver, "get_embeddings"):
            embed_solver = solver
            break
    if embed_solver is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED,
                            detail="No embeddings solver configured for this persona.")

    embeddings = [embed_solver.get_embeddings(t) for t in request.texts]
    return JSONResponse({
        "id": _new_id(),
        "embeddings": embeddings,
        "texts": request.texts,
        "meta": {"api_version": {"version": "1"}},
    })
