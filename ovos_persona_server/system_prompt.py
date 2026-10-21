"""Client system-prompt handling strategies.

By default the persona-server ignores any ``system`` message in an incoming
OpenAI request and uses only the persona's own ``system_prompt`` (so a hosted
persona keeps its identity).  Some drop-in-OpenAI clients need to supply their
own system prompt, so the behaviour is made configurable through
``system_prompt_strategy``:

- ``ignore``  (default) drop client system messages; the persona's own
  ``system_prompt`` is the only system prompt.  Preserves today's behaviour.
- ``replace`` use the client's system message(s) instead of the persona's;
  fall back to the persona's when the client sends none.
- ``append``  persona's ``system_prompt`` first, then the client's system
  message(s) appended after it (persona identity as the base, client adds
  context), joined by a blank line.

The strategy is resolved from the persona config
(``"system_prompt_strategy"``) or the ``PERSONA_SYSTEM_PROMPT_STRATEGY``
environment variable, defaulting to ``ignore``.  It is applied once, on the
incoming message list, so it holds for every downstream path (the text
``Persona.chat`` path and, when present, the OpenAI tools passthrough).
"""

import os
from typing import Any, Dict, List, Optional

VALID_STRATEGIES = ("ignore", "replace", "append")
DEFAULT_STRATEGY = "ignore"


def _flatten_text(content: Any) -> str:
    """Coerce OpenAI message content (str | content-parts | None) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [(p.get("text") or "") if isinstance(p, dict) else str(p) for p in content]
        return " ".join(p for p in parts if p)
    return str(content)


def resolve_strategy(persona: Any) -> str:
    """Resolve the effective strategy for a persona.

    Precedence: the persona config's ``system_prompt_strategy`` key, then the
    ``PERSONA_SYSTEM_PROMPT_STRATEGY`` environment variable, then ``ignore``.
    An unrecognised value falls back to ``ignore``.
    """
    cfg = getattr(persona, "config", None) or {}
    value = cfg.get("system_prompt_strategy") if isinstance(cfg, dict) else None
    value = value or os.environ.get("PERSONA_SYSTEM_PROMPT_STRATEGY")
    value = (value or DEFAULT_STRATEGY).strip().lower()
    return value if value in VALID_STRATEGIES else DEFAULT_STRATEGY


def persona_system_prompt(persona: Any) -> Optional[str]:
    """Best-effort read of the persona's own system prompt.

    Prefers the primary solver's ``system_prompt`` attribute (where the loaded
    OpenAI-style solver keeps it, with defaults resolved), falling back to a
    ``system_prompt`` key on the persona config.
    """
    try:
        modules = persona.solvers.modules
        if isinstance(modules, (list, tuple)) and modules:
            sp = getattr(modules[0], "system_prompt", None)
            if isinstance(sp, str) and sp.strip():
                return sp
    except Exception:
        pass
    cfg = getattr(persona, "config", None) or {}
    if isinstance(cfg, dict):
        sp = cfg.get("system_prompt")
        if isinstance(sp, str) and sp.strip():
            return sp
    return None


def apply_system_prompt_strategy(
        persona: Any,
        messages: List[Dict[str, Any]],
        strategy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Rewrite the message list so the leading system prompt reflects the strategy.

    Any client-supplied ``system`` messages are pulled out (and concatenated,
    in order, when there is more than one).  The resolved system prompt is
    placed as a single leading system message and the non-system turns follow
    unchanged.  When the resolved prompt is empty no system message is added
    (leaving the downstream solver free to inject its own default, preserving
    current behaviour).

    Args:
        persona: The loaded persona (source of its own system prompt / config).
        messages: Incoming OpenAI message dicts.
        strategy: Override; when ``None`` it is resolved from the persona.

    Returns:
        A new message list with the system prompt applied per the strategy.
    """
    strategy = (strategy or resolve_strategy(persona)).strip().lower()
    if strategy not in VALID_STRATEGIES:
        strategy = DEFAULT_STRATEGY

    persona_sp = persona_system_prompt(persona)

    client_systems: List[str] = []
    body: List[Dict[str, Any]] = []
    for m in messages:
        role = getattr(m.get("role"), "value", m.get("role"))
        if role == "system":
            text = _flatten_text(m.get("content"))
            if text:
                client_systems.append(text)
        else:
            body.append(m)
    client_sp = "\n\n".join(client_systems) or None

    if strategy == "replace":
        effective = client_sp or persona_sp
    elif strategy == "append":
        effective = "\n\n".join(p for p in (persona_sp, client_sp) if p) or None
    else:  # ignore
        effective = persona_sp

    if effective:
        return [{"role": "system", "content": effective}] + body
    return body
