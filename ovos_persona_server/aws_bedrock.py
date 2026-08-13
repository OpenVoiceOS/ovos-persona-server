# Licensed under the Apache License, Version 2.0
"""AWS Bedrock-compatible API endpoints."""
import base64
import json
import random
import string
import struct
import time
import zlib
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from ovos_persona import Persona
from pydantic import BaseModel, Field

from ovos_persona_server.embeddings import embed_texts, get_embeddings_backend
from ovos_persona_server.persona import get_default_persona, run_chat, run_stream

bedrock_router = APIRouter(prefix="/bedrock/model", tags=["aws-bedrock"])

EVENTSTREAM_MEDIA_TYPE = "application/vnd.amazon.eventstream"


def _new_id() -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=32))


def _encode_eventstream_message(payload: bytes, headers: Dict[str, str]) -> bytes:
    """Encode one AWS event-stream (vnd.amazon.eventstream) frame.

    The binary framing is what botocore's response stream parser expects, so
    boto3's ``invoke_model_with_response_stream`` can consume the output. Each
    frame is ``prelude | prelude_crc | headers | payload | message_crc`` with
    CRC32 checksums over the prelude and the whole message.

    Args:
        payload: Raw payload bytes for the frame.
        headers: String-valued event-stream headers (e.g. ``:event-type``).

    Returns:
        The encoded binary frame.
    """
    encoded_headers = b""
    for name, value in headers.items():
        name_bytes = name.encode("utf-8")
        value_bytes = value.encode("utf-8")
        # header: name_len(1) | name | value_type(1, 7=string) | value_len(2) | value
        encoded_headers += struct.pack("B", len(name_bytes)) + name_bytes
        encoded_headers += struct.pack("B", 7) + struct.pack(">H", len(value_bytes)) + value_bytes

    total_length = 4 + 4 + 4 + len(encoded_headers) + len(payload) + 4
    prelude = struct.pack(">II", total_length, len(encoded_headers))
    prelude_crc = struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
    message = prelude + prelude_crc + encoded_headers + payload
    message_crc = struct.pack(">I", zlib.crc32(message) & 0xFFFFFFFF)
    return message + message_crc


def _bedrock_chunk_frame(chunk: Dict[str, Any]) -> bytes:
    """Wrap a model-specific chunk dict as a Bedrock ``chunk`` event frame.

    Bedrock streams each chunk as an event whose JSON payload carries the
    base64-encoded, model-specific chunk under a ``bytes`` key.

    Args:
        chunk: The model-specific chunk body (decoded by the client).

    Returns:
        The encoded event-stream frame.
    """
    payload = json.dumps(
        {"bytes": base64.b64encode(json.dumps(chunk).encode("utf-8")).decode("utf-8")}
    ).encode("utf-8")
    return _encode_eventstream_message(
        payload,
        {":event-type": "chunk", ":content-type": "application/json", ":message-type": "event"},
    )


def _is_embedding_model(model_id: str) -> bool:
    """Return True if ``model_id`` names a Bedrock embedding model.

    Covers Amazon Titan (``amazon.titan-embed-*``) and Cohere
    (``cohere.embed-*``) embedding model families.

    Args:
        model_id: Bedrock model identifier from the request path.

    Returns:
        Whether the model produces embeddings rather than text.
    """
    return "embed" in model_id.lower()


def _extract_embed_texts(body: Dict[str, Any], model_id: str) -> List[str]:
    """Extract the input texts from a Bedrock embedding invoke body.

    Handles Cohere (``texts``) and Amazon Titan (``inputText``) request shapes.

    Args:
        body: Parsed JSON request body.
        model_id: Bedrock model identifier (for format detection).

    Returns:
        List of input strings to embed.
    """
    if isinstance(body.get("texts"), list):  # Cohere embed
        return [str(t) for t in body["texts"]]
    if "inputText" in body:  # Amazon Titan embed
        return [str(body["inputText"])]
    if "input" in body:
        value = body["input"]
        return [str(v) for v in value] if isinstance(value, list) else [str(value)]
    return [json.dumps(body)]


def _build_embedding_response(vectors: List[List[float]], model_id: str,
                              texts: List[str]) -> Dict[str, Any]:
    """Build a model-specific Bedrock embedding response body.

    Args:
        vectors: Embedding vectors, one per input text.
        model_id: Bedrock model ID for response-format selection.
        texts: Original input texts (echoed for the Cohere format).

    Returns:
        Dict matching the expected response format for the model family.
    """
    mid = model_id.lower()
    if "cohere.embed" in mid:
        return {
            "id": _new_id(),
            "embeddings": vectors,
            "texts": texts,
            "response_type": "embeddings_floats",
        }
    # Amazon Titan returns a single embedding per invoke.
    return {
        "embedding": vectors[0] if vectors else [],
        "inputTextTokenCount": sum(len(t.split()) for t in texts),
    }


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
    if _is_embedding_model(model_id):
        embedder = await get_embeddings_backend()
        texts = _extract_embed_texts(body, model_id)
        vectors = embed_texts(embedder, texts)
        return JSONResponse(_build_embedding_response(vectors, model_id, texts))
    messages = _extract_messages(body, model_id)
    text = run_chat(persona, messages)
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

    async def _stream() -> AsyncGenerator[bytes, None]:
        accumulated = []
        try:
            for chunk in run_stream(persona, messages):
                if chunk:
                    accumulated.append(chunk)
                    yield _bedrock_chunk_frame(
                        {"outputText": chunk, "index": 0, "totalOutputTextTokenCount": None}
                    )
        except Exception as exc:
            yield _bedrock_chunk_frame({"error": str(exc)})
            return
        full_text = "".join(accumulated)
        yield _bedrock_chunk_frame({
            "outputText": "",
            "index": 0,
            "totalOutputTextTokenCount": len(full_text.split()),
            "completionReason": "FINISH",
        })

    return StreamingResponse(_stream(), media_type=EVENTSTREAM_MEDIA_TYPE)


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

    text = run_chat(persona, messages)
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
