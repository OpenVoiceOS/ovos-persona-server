# Licensed under the Apache License, Version 2.0
"""HuggingFace Text Generation Inference (TGI)-compatible endpoints."""
import json
from typing import AsyncGenerator, Optional, Union

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from ovos_persona import Persona
from pydantic import BaseModel, Field

from ovos_persona_server.persona import get_default_persona

tgi_router = APIRouter(prefix="/tgi", tags=["huggingface-tgi"])


class TGIParameters(BaseModel):
    """Generation parameters for TGI requests."""
    max_new_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, gt=0)
    do_sample: Optional[bool] = None
    repetition_penalty: Optional[float] = Field(default=None, gt=0.0)
    seed: Optional[int] = None
    stop: Optional[list] = None


class TGIRequest(BaseModel):
    """Request body for POST /generate and POST /generate_stream."""
    inputs: str = Field(..., min_length=1)
    parameters: Optional[TGIParameters] = None
    stream: Optional[bool] = None


class TGIToken(BaseModel):
    """A single generated token in streaming response."""
    id: int = Field(..., ge=0)
    text: str
    logprob: float
    special: bool


class TGIDetails(BaseModel):
    """Generation details in TGI response."""
    finish_reason: str
    generated_tokens: int = Field(..., ge=0)


class TGIResponse(BaseModel):
    """Response for POST /generate."""
    generated_text: str
    details: Optional[TGIDetails] = None


@tgi_router.post("/generate", response_model=TGIResponse)
async def generate(
        request: TGIRequest,
        persona: Persona = Depends(get_default_persona),
) -> JSONResponse:
    """Generate text (HuggingFace TGI-compatible).

    Args:
        request: TGI generate request with inputs string.
        persona: Injected persona instance.

    Returns:
        TGIResponse with generated_text.
    """
    messages = [{"role": "user", "content": request.inputs}]
    text = persona.chat(messages)
    return JSONResponse(TGIResponse(
        generated_text=text or "",
        details=TGIDetails(finish_reason="eos_token", generated_tokens=len((text or "").split())),
    ).model_dump())


@tgi_router.post("/generate_stream", response_model=None)
async def generate_stream(
        request: TGIRequest,
        persona: Persona = Depends(get_default_persona),
) -> StreamingResponse:
    """Stream generated text (HuggingFace TGI-compatible).

    Args:
        request: TGI generate request with inputs string.
        persona: Injected persona instance.

    Returns:
        SSE stream of TGI token events.
    """
    messages = [{"role": "user", "content": request.inputs}]

    async def _stream() -> AsyncGenerator[str, None]:
        accumulated = []
        token_count = 0
        try:
            for chunk in persona.stream(messages):
                if chunk:
                    accumulated.append(chunk)
                    token_count += 1
                    event = json.dumps({
                        "token": {"id": token_count, "text": chunk, "logprob": -1.0, "special": False},
                        "generated_text": None,
                        "details": None,
                    })
                    yield f"data:{event}\n\n"
        except Exception as exc:
            yield f"data:{json.dumps({'error': str(exc)})}\n\n"
            return

        full_text = "".join(accumulated)
        final = json.dumps({
            "token": {"id": 0, "text": "", "logprob": 0.0, "special": True},
            "generated_text": full_text,
            "details": {"finish_reason": "eos_token", "generated_tokens": token_count},
        })
        yield f"data:{final}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@tgi_router.get("/info", response_model=None)
async def info(persona: Persona = Depends(get_default_persona)) -> JSONResponse:
    """Return model info (HuggingFace TGI-compatible).

    Args:
        persona: Injected persona instance.

    Returns:
        TGI-format model info dict.
    """
    return JSONResponse({
        "model_id": persona.name,
        "model_dtype": "float16",
        "model_device_type": "cpu",
        "model_pipeline_tag": "text-generation",
        "max_concurrent_requests": 128,
        "max_best_of": 1,
        "max_stop_sequences": 4,
        "max_input_length": 4096,
        "max_total_tokens": 8192,
        "waiting_served_ratio": 1.2,
        "cache_max_total_tokens": 1000000,
        "max_batch_total_tokens": 32000,
        "validation_workers": 1,
        "version": "2.0.0",
        "sha": "placeholder",
    })


@tgi_router.get("/health", response_model=None)
async def health() -> JSONResponse:
    """Health check (HuggingFace TGI-compatible).

    Returns:
        Empty JSON object with 200 status.
    """
    return JSONResponse({})
