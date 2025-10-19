"""
FastAPI router for OpenAI-compatible embeddings endpoints.

This module provides an API endpoint for generating text embeddings
using configured OVOS text embedding plugins.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, Depends, status, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from ovos_plugin_manager.embeddings import load_text_embeddings_plugin
from ovos_plugin_manager.templates.embeddings import ImageEmbedder, TextEmbedder

from ovos_persona_server.config import settings
from ovos_persona_server.schemas.openai_embeddings import CreateEmbeddingResponse, CreateEmbeddingRequest, Embedding, \
    Usage


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Ensure text embeddings are loaded on application startup and yield control for request handling.
    
    This lifespan context manager loads the configured text embeddings plugin before the app starts serving requests and then yields to allow normal operation; no shutdown actions are performed.
    """
    await get_text_embeddings()
    # await get_image_embeddings() # Uncomment if image embeddings are to be used
    yield
    # No specific shutdown logic needed for these dependencies currently


embeddings_router = APIRouter(prefix="/v1", tags=["openai"], lifespan=lifespan)

# Dependency injection
text_embeddings: Optional[TextEmbedder] = None
image_embeddings: Optional[ImageEmbedder] = None


async def get_text_embeddings() -> TextEmbedder:
    """
    Load and return the configured text embeddings plugin, caching it for reuse.
    
    Acts as a FastAPI dependency that ensures a single TextEmbedder instance is loaded and reused across requests.
    
    Returns:
        TextEmbedder: The loaded OVOS TextEmbedder instance.
    
    Raises:
        HTTPException: If the text embeddings plugin fails to load or cannot be instantiated.
    """
    global text_embeddings
    if text_embeddings is None:
        try:
            plugin_class = load_text_embeddings_plugin(settings.text_embeddings_plugin)
            text_embeddings = plugin_class(settings.embeddings_config)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Failed to load text embeddings plugin: {e}") from e
    return text_embeddings


async def get_image_embeddings() -> Optional[ImageEmbedder]:
    """
    Return the configured image embeddings plugin, if available.
    
    Returns:
        Optional[ImageEmbedder]: The loaded ImageEmbedder instance, or `None` if no image embedder is configured or the feature is not implemented.
    """
    global image_embeddings
    # TODO: placeholder - implement real image embeddings when ready
    # if image_embeddings is None:
    #    image_embeddings = load_image_embeddings_plugin(settings.image_embeddings_plugin)(cfg)
    return image_embeddings


@embeddings_router.post("/embeddings", response_model=CreateEmbeddingResponse, status_code=status.HTTP_200_OK)
async def create_embeddings(
        request_body: CreateEmbeddingRequest,
        embedder: TextEmbedder = Depends(get_text_embeddings)
) -> JSONResponse:
    """
        Generate embeddings for the provided text input(s) and return an OpenAI-compatible response payload.
        
        Parameters:
            request_body (CreateEmbeddingRequest): Request containing `input` (a string or list of strings) and optional `encoding_format`. `input` must be a string or an array of strings; other formats will trigger a 400 error. `encoding_format="base64"` is not supported and will trigger a 500 error.
        
        Returns:
            JSONResponse: A JSON payload with fields `object`, `data` (list of embeddings with index and vector), `model`, and `usage` (prompt_tokens and total_tokens).
        
        Raises:
            HTTPException: 400 if `input` is not a string or list of strings;
                           500 if embedding generation fails or if an unsupported encoding_format (e.g., "base64") is requested.
        """
    # The Pydantic model handles the validation of `request_body.input` type
    # It can be str, List[str], List[int], or List[List[int]]
    texts_to_embed: List[str] = []

    if isinstance(request_body.input, str):
        texts_to_embed = [request_body.input]
    elif isinstance(request_body.input, list):
        if all(isinstance(item, str) for item in request_body.input):
            texts_to_embed = request_body.input
        else:
            # This case should ideally be caught by Pydantic validation, but as a fallback
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Input must be a string or array of strings.")
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Input must be a string or array of strings.")

    model_name: str = embedder.config.get("model",
                                          "default")  # TODO - dynamic support (load multiple embeddings plugins)

    prompt_tokens: int = sum(len(text.split()) for text in texts_to_embed) if texts_to_embed else 0
    total_tokens: int = prompt_tokens

    try:
        vectors: List[List[float]] = [embedder.get_embeddings(t) for t in texts_to_embed]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to generate embeddings: {e}") from e

    # TODO - b64
    if request_body.encoding_format == "base64":
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to generate embeddings: base64 support not yet implemented")

    embeddings_data: List[Embedding] = []
    for i, vector in enumerate(vectors):
        embeddings_data.append(Embedding(
            object="embedding",
            index=i,
            embedding=vector
        ))

    usage_obj: Usage = Usage(
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens
    )

    response_obj = CreateEmbeddingResponse(
        object="list",
        data=embeddings_data,
        model=embedder.config.get("model", "server-default"),
        usage=usage_obj
    )
    return JSONResponse(content=response_obj.model_dump(exclude_unset=True))