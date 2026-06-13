"""
Shared, swappable text-embeddings backend for the persona server.

A single embeddings service backs every vendor router (OpenAI, Ollama, …) and
the vector-store search path, mirroring how inference is backed by one shared
persona. The service is resolved by :func:`get_embeddings_backend` and is fully
configurable — point it at any embeddings provider through the
``TEXT_EMBEDDINGS_PLUGIN`` / ``EMBEDDINGS_URL`` / ``EMBEDDINGS_MODEL`` settings
(any OVOS text-embeddings plugin, local or remote). When no plugin is available
it falls back to a persona solver exposing ``get_embeddings``.
"""
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, List, Optional

from fastapi import FastAPI, Depends, status, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from ovos_plugin_manager.embeddings import load_text_embeddings_plugin
from ovos_plugin_manager.templates.embeddings import ImageEmbedder, TextEmbedder

from ovos_persona_server.config import settings
from ovos_persona_server.schemas.openai_embeddings import CreateEmbeddingResponse, CreateEmbeddingRequest, Embedding, \
    Usage

LOG = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages the lifespan of the FastAPI application, ensuring text embeddings are loaded.
    """
    await get_text_embeddings()
    # await get_image_embeddings() # Uncomment if image embeddings are to be used
    yield
    # No specific shutdown logic needed for these dependencies currently


embeddings_router = APIRouter(prefix="/openai/v1", tags=["openai"])

# Dependency injection
text_embeddings: Optional[TextEmbedder] = None
image_embeddings: Optional[ImageEmbedder] = None


async def get_text_embeddings() -> TextEmbedder:
    """
    Asynchronously loads and returns the configured text embeddings plugin.

    This function acts as a dependency for FastAPI endpoints, ensuring
    the text embedder is loaded once and reused across requests.

    Returns:
        TextEmbedder: The loaded OVOS TextEmbedder instance.

    Raises:
        HTTPException: If the text embeddings plugin fails to load.
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
    Asynchronously loads and returns the configured image embeddings plugin.

    NOTE: This is currently a placeholder and does not load a real image embedder.

    Returns:
        Optional[ImageEmbedder]: A placeholder OVOS ImageEmbedder instance, or None if not implemented.

    Raises:
        HTTPException: If the image embeddings plugin fails to load (currently not implemented).
    """
    global image_embeddings
    # TODO: placeholder - implement real image embeddings when ready
    # if image_embeddings is None:
    #    image_embeddings = load_image_embeddings_plugin(settings.image_embeddings_plugin)(cfg)
    return image_embeddings


class _SolverEmbedder:
    """Adapt a persona solver exposing ``get_embeddings`` to the embedder interface.

    Lets a persona that bundles an embeddings-capable solver act as the shared
    embeddings backend when no dedicated embeddings plugin is configured.
    """

    def __init__(self, solver: Any) -> None:
        self._solver = solver
        self.config: dict = {}

    def get_embeddings(self, text: str) -> List[float]:
        return self._solver.get_embeddings(text)


def _persona_solver_embedder() -> Optional[_SolverEmbedder]:
    """Return an embedder backed by the default persona's first embeddings solver, if any."""
    from ovos_persona_server.persona import default_persona

    if default_persona is None:
        return None
    for solver in default_persona.solvers.loaded_modules.values():
        if hasattr(solver, "get_embeddings"):
            return _SolverEmbedder(solver)
    return None


async def get_embeddings_backend() -> TextEmbedder:
    """Return the shared, swappable text-embeddings backend used by every router.

    This is the embeddings analogue of :func:`get_default_persona`: a single
    service that every vendor endpoint (OpenAI, Ollama, …) and the vector-store
    search path delegate to, so swapping the embeddings provider in one place
    changes it everywhere.

    Resolution order:

    1. The configured OVOS text-embeddings plugin
       (``settings.text_embeddings_plugin``) — point it at any embeddings
       service via ``EMBEDDINGS_URL`` / ``EMBEDDINGS_MODEL`` / ``EMBEDDINGS_KEY``.
    2. A persona solver exposing ``get_embeddings`` — used when no embeddings
       plugin is installed or it fails to load.

    Override it per-app with ``app.dependency_overrides[get_embeddings_backend]``.

    Raises:
        HTTPException: 501 when no embeddings backend can be resolved.
    """
    global text_embeddings
    if text_embeddings is None:
        try:
            plugin_class = load_text_embeddings_plugin(settings.text_embeddings_plugin)
            text_embeddings = plugin_class(settings.embeddings_config)
        except Exception as e:
            LOG.warning("text embeddings plugin '%s' unavailable (%s); "
                        "falling back to persona solver", settings.text_embeddings_plugin, e)
            fallback = _persona_solver_embedder()
            if fallback is not None:
                return fallback
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="No embeddings backend available: configure TEXT_EMBEDDINGS_PLUGIN "
                       "(optionally EMBEDDINGS_URL/EMBEDDINGS_MODEL) or load a persona solver "
                       "exposing get_embeddings.",
            ) from e
    return text_embeddings


def embed_texts(backend: TextEmbedder, texts: List[str]) -> List[List[float]]:
    """Embed ``texts`` with ``backend`` and coerce each vector to ``list[float]``.

    Args:
        backend: Any object exposing ``get_embeddings(text) -> sequence[float]``.
        texts: The strings to embed.

    Returns:
        One float vector per input text.

    Raises:
        HTTPException: 500 if the backend raises while generating embeddings.
    """
    try:
        return [[float(x) for x in backend.get_embeddings(t)] for t in texts]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to generate embeddings: {e}") from e


def backend_model_name(backend: TextEmbedder, requested: Optional[str] = None) -> str:
    """Best-effort model identifier for an embeddings response."""
    if requested:
        return requested
    cfg = getattr(backend, "config", None) or {}
    return cfg.get("model", "server-default")


@embeddings_router.post("/embeddings", response_model=CreateEmbeddingResponse, status_code=status.HTTP_200_OK)
async def create_embeddings(
        request_body: CreateEmbeddingRequest,
        embedder: TextEmbedder = Depends(get_embeddings_backend)
) -> JSONResponse:
    """
    Generates embeddings for the given text input(s).

    Args:
        request_body (CreateEmbeddingRequest): The request body containing
                                                the input text(s) and model.
        embedder (TextEmbedder): The text embedder instance, injected by FastAPI's dependency.

    Returns:
        JSONResponse: A JSON response containing the generated embeddings and usage information.

    Raises:
        HTTPException: If the input format is invalid or embedding generation fails.
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
