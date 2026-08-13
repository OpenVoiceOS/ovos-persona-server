# Licensed under the Apache License, Version 2.0
"""Anthropic Claude-compatible API endpoints."""
import json
import random
import string
import time
from typing import AsyncGenerator, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from ovos_persona import Persona

from ovos_persona_server.persona import get_default_persona, run_chat, run_stream
from ovos_persona_server.schemas.anthropic import (
    AnthropicContentBlock,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicUsage,
)

anthropic_router = APIRouter(prefix="/anthropic/v1", tags=["anthropic"])


def _normalise_messages(request: AnthropicRequest) -> List[Dict[str, str]]:
    """Normalise Anthropic messages to persona-compatible format.

    Prepends system message if provided. Flattens content block lists to strings.

    Args:
        request: Anthropic API request.

    Returns:
        List of message dicts with 'role' and 'content' string keys.
    """
    messages: List[Dict[str, str]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    for msg in request.messages:
        if isinstance(msg.content, str):
            content = msg.content
        else:
            content = " ".join(block.text for block in msg.content)
        messages.append({"role": msg.role, "content": content})
    return messages


@anthropic_router.post("/messages", response_model=None)
async def create_message(
        request: AnthropicRequest,
        persona: Persona = Depends(get_default_persona),
        x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> Union[JSONResponse, StreamingResponse]:
    """Create a message (Anthropic Claude-compatible).

    Args:
        request: Anthropic messages request body.
        persona: Injected persona instance.
        x_api_key: Anthropic API key header (accepted, ignored).

    Returns:
        JSON response or SSE stream in Anthropic format.
    """
    messages = _normalise_messages(request)
    msg_id = "msg_" + "".join(random.choices(string.ascii_letters + string.digits, k=24))

    if not request.stream:
        try:
            content = run_chat(persona, messages)
            return JSONResponse(AnthropicResponse(
                id=msg_id,
                content=[AnthropicContentBlock(type="text", text=content)],
                model=request.model,
                usage=AnthropicUsage(
                    input_tokens=sum(len(m["content"].split()) for m in messages),
                    output_tokens=len(content.split()),
                ),
            ).model_dump())
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Persona chat failed: {exc}") from exc

    async def _stream() -> AsyncGenerator[str, None]:
        """Yield SSE events in Anthropic streaming format."""
        yield (
            f"event: message_start\ndata: {{\"type\":\"message_start\","
            f"\"message\":{{\"id\":\"{msg_id}\",\"type\":\"message\","
            f"\"role\":\"assistant\",\"content\":[],"
            f"\"model\":\"{request.model}\",\"stop_reason\":null,"
            f"\"usage\":{{\"input_tokens\":0,\"output_tokens\":0}}}}}}\n\n"
        )
        yield (
            "event: content_block_start\ndata: {\"type\":\"content_block_start\","
            "\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
        )

        try:
            for chunk in run_stream(persona, messages):
                if chunk:
                    delta = json.dumps({
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": chunk},
                    })
                    yield f"event: content_block_delta\ndata: {delta}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {{\"type\":\"error\",\"error\":{{\"message\":\"{exc}\"}}}}\n\n"
            return

        yield "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
        yield (
            "event: message_delta\ndata: {\"type\":\"message_delta\","
            "\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\n"
        )
        yield "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
