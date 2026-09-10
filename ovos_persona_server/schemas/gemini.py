# Licensed under the Apache License, Version 2.0
"""Pydantic schemas for Google Gemini API compatibility."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class GeminiPart(BaseModel):
    """A single text part in a Gemini content block."""

    text: str


class GeminiContent(BaseModel):
    """A content block in a Gemini conversation."""

    role: Literal["user", "model"]
    parts: List[GeminiPart] = Field(..., min_length=1)


class GeminiSystemInstruction(BaseModel):
    """System instruction for Gemini request."""

    parts: List[GeminiPart] = Field(..., min_length=1)


class GeminiGenerationConfig(BaseModel):
    """Generation configuration for Gemini request."""

    maxOutputTokens: Optional[int] = Field(default=None, gt=0)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    topP: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    topK: Optional[int] = Field(default=None, gt=0)


class GeminiRequest(BaseModel):
    """Request body for Gemini generateContent."""

    contents: List[GeminiContent] = Field(..., min_length=1)
    systemInstruction: Optional[GeminiSystemInstruction] = None
    generationConfig: Optional[GeminiGenerationConfig] = None


class GeminiCandidate(BaseModel):
    """A single generated response candidate."""

    content: GeminiContent
    finishReason: Literal["STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"] = "STOP"
    index: int = Field(default=0, ge=0)


class GeminiResponse(BaseModel):
    """Response for Gemini generateContent."""

    candidates: List[GeminiCandidate] = Field(..., min_length=1)


class GeminiEmbedContent(BaseModel):
    """Content block for an embedContent request (role optional/ignored)."""

    parts: List[GeminiPart] = Field(..., min_length=1)


class GeminiEmbedContentRequest(BaseModel):
    """Request body for models/{model}:embedContent."""

    content: GeminiEmbedContent
    model: Optional[str] = None
    taskType: Optional[str] = None
    title: Optional[str] = None
    outputDimensionality: Optional[int] = Field(default=None, gt=0)


class GeminiContentEmbedding(BaseModel):
    """A single embedding vector in a Gemini embeddings response."""

    values: List[float]


class GeminiEmbedContentResponse(BaseModel):
    """Response for models/{model}:embedContent."""

    embedding: GeminiContentEmbedding


class GeminiBatchEmbedRequestItem(BaseModel):
    """One sub-request inside a batchEmbedContents call."""

    content: GeminiEmbedContent
    model: Optional[str] = None
    taskType: Optional[str] = None
    title: Optional[str] = None
    outputDimensionality: Optional[int] = Field(default=None, gt=0)


class GeminiBatchEmbedContentsRequest(BaseModel):
    """Request body for models/{model}:batchEmbedContents."""

    requests: List[GeminiBatchEmbedRequestItem] = Field(..., min_length=1)


class GeminiBatchEmbedContentsResponse(BaseModel):
    """Response for models/{model}:batchEmbedContents."""

    embeddings: List[GeminiContentEmbedding] = Field(..., min_length=1)
