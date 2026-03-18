# Licensed under the Apache License, Version 2.0
"""AWS Bedrock-compatible API endpoints."""
import json
import random
import string
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from ovos_persona import Persona
from pydantic import BaseModel, Field

from ovos_persona_server.persona import get_default_persona

bedrock_router = APIRouter(prefix="/model", tags=["aws-bedrock"])


def _new_id() -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=32))


def _extract_messages(body: Dict[str, Any], model_id: str) -> List[Dict[str, str]]:
    """Extract persona-compatible messages from a Bedrock invoke body.

    Handles Anthropic, Llama, Titan, Cohere, and Bedrock Converse formats.

    Args:
        body: Parsed JSON request body.
        model_id: Bedrock model identifier for format detection.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    mid = model_id.lower()

    # Bedrock Converse API format
    if "messages" in body and isinstance(body["messages"], list):
        msgs = body["messages"]
        result = []
        if body.get("system"):
            sys_content = body["system"]
            if isinstance(sys_content, list):
                sys_text = " ".join(b.get("text", "") for b in sys_content)
            else:
                sys_text = str(sys_content)
            result.append({"role": "system", "content": sys_text})
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            # Converse format: content is a list of content blocks
            if isinstance(content, list):
                text = " ".join(
                    block.get("text", block.get("content", ""))
                    if isinstance(block, dict) else str(block)
                    for block in content
                )
            else:
                text = str(content)
            result.append({"role": role, "content": text})
        return result

    # Llama / generic prompt
    if "prompt" in body:
        return [{"role": "user", "content": str(body["prompt"])}]

    # Titan
    if "inputText" in body:
        return [{"role": "user", "content": str(body["inputText"])}]

    # Cohere Command
    if "message" in body:
        msgs = []
        for h in body.get("chat_history", []):
            role_map = {"USER": "user", "CHATBOT": "assistant"}
            msgs.append({"role": role_map.get(h.get("role", "USER"), "user"),
                          "content": h.get("message", "")})
        msgs.append({"role": "user", "content": str(body["message"])})
        return msgs

    return [{"role": "user", "content": json.dumps(body)}]


def _build_response(text: str, model_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Build a model-specific response body.

    Args:
        text: Generated text from persona.
        model_id: Bedrock model ID for format selection.
        body: Original request body.

    Returns:
        Dict matching the expected response format for the model family.
    """
    mid = model_id.lower()
    if "anthropic.claude" in mid:
        return {
            "id": _new_id(),
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": model_id,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": len(text.split())},
        }
    if "meta.llama" in mid:
        return {
            "generation": text,
            "prompt_token_count": 0,
            "generation_token_count": len(text.split()),
            "stop_reason": "stop",
        }
    if "amazon.titan" in mid:
        return {
            "inputTextTokenCount": 0,
            "results": [{"tokenCount": len(text.split()), "outputText": text, "completionReason": "FINISH"}],
        }
    if "cohere.command" in mid:
        return {
            "id": _new_id(),
            "generations": [{"id": _new_id(), "text": text, "finish_reason": "COMPLETE"}],
        }
    # Generic fallback
    return {"outputText": text, "generated_text": text}


@bedrock_router.post("/{model_id}/invoke", response_model=None)
async def invoke(
        model_id: str,
        request: Request,
        authorization: Optional[str] = Header(default=None),
        persona: Persona = Depends(get_default_persona),
) -> JSONResponse:
    """Invoke a Bedrock model (AWS Bedrock-compatible).

    Accepts any Bedrock invoke body and maps to persona.chat().

    Args:
        model_id: Bedrock model ID (path param, used for response formatting).
        request: Raw request — body is JSON.
        authorization: AWS SigV4 auth header (accepted, ignored).
        persona: Injected persona instance.

    Returns:
        Model-specific JSON response.
    """
    body = await request.json()
    messages = _extract_messages(body, model_id)
    text = persona.chat(messages)
    return JSONResponse(_build_response(text or "", model_id, body))


@bedrock_router.post("/{model_id}/invoke-with-response-stream", response_model=None)
async def invoke_stream(
        model_id: str,
        request: Request,
        authorization: Optional[str] = Header(default=None),
        persona: Persona = Depends(get_default_persona),
) -> StreamingResponse:
    """Invoke a Bedrock model with streaming (AWS Bedrock-compatible).

    Args:
        model_id: Bedrock model ID.
        request: Raw request — body is JSON.
        authorization: AWS SigV4 auth header (accepted, ignored).
        persona: Injected persona instance.

    Returns:
        SSE stream of JSON chunks.
    """
    body = await request.json()
    messages = _extract_messages(body, model_id)

    async def _stream() -> AsyncGenerator[str, None]:
        accumulated = []
        try:
            for chunk in persona.stream(messages):
                if chunk:
                    accumulated.append(chunk)
                    yield f"data:{json.dumps({'outputText': chunk, 'index': 0, 'totalOutputTextTokenCount': None})}\n\n"
        except Exception as exc:
            yield f"data:{json.dumps({'error': str(exc)})}\n\n"
            return
        full_text = "".join(accumulated)
        yield f"data:{json.dumps({'outputText': '', 'index': 0, 'totalOutputTextTokenCount': len(full_text.split()), 'completionReason': 'FINISH'})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


class BedrockConverseContentBlock(BaseModel):
    """A content block in a Bedrock Converse message."""
    text: str


class BedrockConverseMessage(BaseModel):
    """A message in a Bedrock Converse request."""
    role: str
    content: List[BedrockConverseContentBlock] = Field(..., min_length=1)


class BedrockConverseRequest(BaseModel):
    """Request body for POST /model/{model_id}/converse."""
    messages: List[BedrockConverseMessage] = Field(..., min_length=1)
    system: Optional[List[BedrockConverseContentBlock]] = None
    inferenceConfig: Optional[Dict[str, Any]] = None


@bedrock_router.post("/{model_id}/converse", response_model=None)
async def converse(
        model_id: str,
        request: BedrockConverseRequest,
        authorization: Optional[str] = Header(default=None),
        persona: Persona = Depends(get_default_persona),
) -> JSONResponse:
    """Invoke model via Bedrock Converse API (AWS Bedrock-compatible).

    Provides a unified message format across all model families.

    Args:
        model_id: Bedrock model ID (used for response formatting).
        request: Converse API request with messages and optional system.
        authorization: AWS SigV4 auth header (accepted, ignored).
        persona: Injected persona instance.

    Returns:
        Bedrock Converse response format.
    """
    messages: List[Dict[str, str]] = []
    if request.system:
        sys_text = " ".join(b.text for b in request.system)
        messages.append({"role": "system", "content": sys_text})
    for m in request.messages:
        text = " ".join(b.text for b in m.content)
        messages.append({"role": m.role, "content": text})

    text = persona.chat(messages)
    return JSONResponse({
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text or ""}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": sum(len(m["content"].split()) for m in messages),
            "outputTokens": len((text or "").split()),
            "totalTokens": sum(len(m["content"].split()) for m in messages) + len((text or "").split()),
        },
    })
