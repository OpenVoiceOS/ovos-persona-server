"""Tool-calling passthrough for OpenAI-backed personas.

The default persona-server path forwards only ``messages`` to
``Persona.chat`` and returns plain text, so an incoming ``tools`` field is
silently dropped and function calling can never work.  This module adds a
narrow passthrough: when a chat-completions request carries a non-empty
``tools`` list *and* the persona's primary solver is an OpenAI-compatible
chat endpoint, the request is proxied straight to that upstream endpoint and
its response is relayed verbatim — including ``tool_calls``,
``finish_reason: tool_calls``, usage and streamed deltas.

Only OpenAI-backed personas can honour tools; for any other persona the
caller keeps the existing text-only path (tools remain unsupported there,
which is inherent to those solvers).
"""

import json
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# (connect, read) timeouts in seconds.  A read timeout is mandatory so a slow
# or dead upstream can never hang the request forever.
DEFAULT_TIMEOUT: Tuple[int, int] = (10, 300)


def get_openai_chat_solver(persona: Any) -> Optional[Any]:
    """Return the persona's primary solver if it is an OpenAI chat endpoint.

    The passthrough is only taken when the *first* (primary) solver in the
    persona's sort order is an OpenAI-compatible chat solver — one exposing an
    ``api_url`` (already suffixed with ``/chat/completions``), an ``engine``
    (model name) and a ``key``.  Any other primary solver returns ``None`` so
    the caller falls back to the text-only path.

    Args:
        persona: The loaded persona instance.

    Returns:
        The primary solver object, or ``None`` if it is not OpenAI-backed.
    """
    try:
        modules = persona.solvers.modules
    except Exception:
        return None
    # Guard against mocks / non-list module containers: only a real ordered
    # list of solvers is trusted here.
    if not isinstance(modules, (list, tuple)) or not modules:
        return None
    primary = modules[0]
    if all(hasattr(primary, attr) for attr in ("api_url", "engine", "key")):
        return primary
    return None


def _jsonable(value: Any) -> Any:
    """Convert pydantic models / enums into plain JSON-serialisable data."""
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_unset=True, exclude_none=True)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def build_upstream_payload(
        solver: Any,
        request_body: Any,
        messages: List[Dict[str, Any]],
        stream: bool,
) -> Dict[str, Any]:
    """Build the upstream OpenAI chat-completions request body.

    The persona's ``system_prompt`` is injected as a leading system message
    only when the incoming messages do not already start with one (so a
    caller-supplied system message is never doubled).  ``tools``,
    ``tool_choice`` and ``parallel_tool_calls`` are passed through verbatim
    and the model is taken from the solver config.

    Args:
        solver: The OpenAI chat solver (source of model / system prompt).
        request_body: The parsed incoming chat-completions request.
        messages: The incoming messages as plain dicts (verbatim, so prior
            assistant ``tool_calls`` and ``role: tool`` results survive).
        stream: Whether to request a streamed response.

    Returns:
        A JSON-serialisable payload for the upstream endpoint.
    """
    out_messages: List[Dict[str, Any]] = list(messages)
    system_prompt = getattr(solver, "system_prompt", None)
    if system_prompt and not (out_messages and out_messages[0].get("role") == "system"):
        out_messages = [{"role": "system", "content": system_prompt}] + out_messages

    payload: Dict[str, Any] = {
        "model": solver.engine,
        "messages": out_messages,
        "stream": stream,
    }
    tools = _jsonable(request_body.tools)
    if tools:
        payload["tools"] = tools
    if request_body.tool_choice is not None:
        payload["tool_choice"] = _jsonable(request_body.tool_choice)
    if request_body.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = request_body.parallel_tool_calls
    return payload


def _headers(solver: Any) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if getattr(solver, "key", None):
        headers["Authorization"] = "Bearer " + solver.key
    return headers


def _timeout(solver: Any) -> Any:
    return getattr(solver, "config", {}).get("request_timeout") or DEFAULT_TIMEOUT


def _error_response(message: str, status_code: int = 502) -> JSONResponse:
    """OpenAI-shaped error body (never a hang, never a bare 500)."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "upstream_error", "code": None}},
    )


def _proxy_non_stream(solver: Any, payload: Dict[str, Any]) -> JSONResponse:
    """Proxy a non-streaming request and relay the upstream JSON verbatim."""
    try:
        resp = requests.post(
            solver.api_url,
            headers=_headers(solver),
            data=json.dumps(payload),
            timeout=_timeout(solver),
        )
    except requests.exceptions.Timeout:
        return _error_response(f"upstream request to {solver.api_url} timed out", 504)
    except requests.exceptions.RequestException as e:
        return _error_response(f"upstream request failed: {e}", 502)

    try:
        body = resp.json()
    except ValueError:
        return _error_response(
            f"upstream returned non-JSON response (status {resp.status_code})", 502
        )
    # Relay verbatim, preserving the upstream status code (errors included).
    return JSONResponse(status_code=resp.status_code, content=body)


def _proxy_stream(solver: Any, payload: Dict[str, Any]) -> StreamingResponse:
    """Proxy a streaming request and relay upstream SSE lines verbatim.

    The generator is synchronous so Starlette iterates it in a worker thread,
    keeping blocking network I/O off the event loop.
    """
    def gen() -> Iterator[str]:
        try:
            resp = requests.post(
                solver.api_url,
                headers=_headers(solver),
                data=json.dumps(payload),
                stream=True,
                timeout=_timeout(solver),
            )
        except requests.exceptions.Timeout:
            err = json.dumps({"error": {"message": f"upstream request to {solver.api_url} timed out",
                                        "type": "upstream_error", "code": None}})
            yield f"data: {err}\n\n"
            yield "data: [DONE]\n\n"
            return
        except requests.exceptions.RequestException as e:
            err = json.dumps({"error": {"message": f"upstream request failed: {e}",
                                        "type": "upstream_error", "code": None}})
            yield f"data: {err}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                # Relay upstream SSE lines verbatim (data:, [DONE], comments).
                yield f"{line}\n\n"
        except requests.exceptions.RequestException as e:
            err = json.dumps({"error": {"message": f"upstream stream failed: {e}",
                                        "type": "upstream_error", "code": None}})
            yield f"data: {err}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            resp.close()

    return StreamingResponse(gen(), media_type="text/event-stream")


async def handle_tools_passthrough(
        solver: Any,
        request_body: Any,
        messages: List[Dict[str, Any]],
        stream: bool,
):
    """Proxy a tools request to the OpenAI-backed solver and relay the reply.

    Args:
        solver: The persona's primary OpenAI chat solver.
        request_body: The parsed incoming chat-completions request.
        messages: The incoming messages as plain dicts.
        stream: Whether the client requested a streamed response.

    Returns:
        A ``StreamingResponse`` (stream) or ``JSONResponse`` (non-stream).
        The relayed body keeps the upstream shape, so ``model`` reflects the
        upstream model rather than the persona name.
    """
    payload = build_upstream_payload(solver, request_body, messages, stream)
    if stream:
        return _proxy_stream(solver, payload)
    return await run_in_threadpool(_proxy_non_stream, solver, payload)
