# Licensed under the Apache License, Version 2.0
"""Pydantic schemas for Anthropic Claude API compatibility."""
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class AnthropicContentBlock(BaseModel):
    """A single content block in an Anthropic message."""

    type: Literal["text"] = "text"
    text: str


class AnthropicMessage(BaseModel):
    """A message in an Anthropic conversation."""

    role: Literal["user", "assistant", "system"]
    content: Union[str, List[AnthropicContentBlock]] = Field(..., description="Message content")


class AnthropicRequest(BaseModel):
    """Request body for POST /v1/messages."""

    model: str = Field(..., min_length=1)
    messages: List[AnthropicMessage] = Field(..., min_length=1)
    system: Optional[str] = Field(default=None, min_length=1)
    max_tokens: int = Field(default=1024, gt=0)
    stream: bool = False


class AnthropicUsage(BaseModel):
    """Token usage in Anthropic response."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class AnthropicResponse(BaseModel):
    """Response for POST /v1/messages."""

    id: str = Field(..., min_length=1)
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[AnthropicContentBlock]
    model: str = Field(..., min_length=1)
    stop_reason: str = "end_turn"
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage
