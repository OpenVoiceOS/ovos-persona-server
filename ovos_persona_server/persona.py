"""
Module for managing and serving persona configurations.

This module defines the FastAPI router for persona-related endpoints,
including loading the default persona and providing its status.
"""

import json
from contextlib import asynccontextmanager
from tempfile import NamedTemporaryFile
from typing import AsyncGenerator, Optional, List, Tuple, Union

import requests
from fastapi import FastAPI, File, UploadFile, Depends, APIRouter, HTTPException, status, Body, Response
from ovos_document_chunkers.files.doc import DOCParagraphSplitter
from ovos_document_chunkers.files.docx import DOCxParagraphSplitter
from ovos_document_chunkers.files.markdown import MarkdownParagraphSplitter
from ovos_document_chunkers.files.pdf import PDFParagraphSplitter
from ovos_document_chunkers.files.webpages import HTMLParagraphSplitter
from ovos_document_chunkers.text.paragraphs import WtPParagraphSplitter
from ovos_document_chunkers.text.sentence import WtPSentenceSplitter
from ovos_persona import Persona
from ovos_plugin_manager.solvers import load_tldr_solver_plugin, load_multiple_choice_solver_plugin
from ovos_plugin_manager.templates.solvers import TldrSolver, MultipleChoiceSolver
from ovos_utils.log import LOG

from ovos_persona_server.config import settings
from ovos_persona_server.schemas import Status, SummarizeRequest, RerankRequest, SegmentRequest
from ovos_persona_server.version import VERSION_ALPHA, VERSION_MINOR, VERSION_MAJOR


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages the lifespan of the FastAPI application.

    This context manager ensures that the default persona and necessary
    components are loaded when the application starts and cleans up
    when it shuts down.

    Yields:
        None: Control is yielded to the FastAPI application.
    """
    global wtp_paragraph_splitter, wtp_sentence_splitter, default_persona, reranker_plugin, summarizer_plugin
    wtp_paragraph_splitter = await get_wtp_paragraph_splitter()
    wtp_sentence_splitter = await get_wtp_sentence_splitter(wtp_paragraph_splitter)
    default_persona = await get_default_persona()
    reranker_plugin = await get_reranker_plugin()
    summarizer_plugin = await get_summarizer_plugin()
    yield
    # No specific shutdown logic needed for these dependencies currently


persona_router = APIRouter(prefix="", tags=["OpenVoiceOS"], lifespan=lifespan)

# Dependency injection
default_persona: Optional[Persona] = None
summarizer_plugin: Optional[TldrSolver] = None
reranker_plugin: Optional[MultipleChoiceSolver] = None
wtp_paragraph_splitter: Optional[WtPParagraphSplitter] = None
wtp_sentence_splitter: Optional[WtPSentenceSplitter] = None


async def get_wtp_paragraph_splitter() -> WtPParagraphSplitter:
    """
    Provides a singleton instance of WtPParagraphSplitter.

    This function initializes the WtPParagraphSplitter if it has not
    been created yet and returns the instance.

    Returns:
        WtPParagraphSplitter: The initialized paragraph splitter instance.
    """
    global wtp_paragraph_splitter
    if wtp_paragraph_splitter is None:
        wtp_paragraph_splitter = WtPParagraphSplitter(
            config={"model": settings.WtP_model,
                    "use_cuda": settings.WtP_cuda})
    return wtp_paragraph_splitter


async def get_wtp_sentence_splitter(
        wtp_paragraph_splitter: Optional[WtPParagraphSplitter] = None
) -> WtPSentenceSplitter:
    """
    Provides a singleton instance of WtPSentenceSplitter, sharing the WtP model.

    This function initializes the WtPSentenceSplitter if it has not
    been created yet, using the provided WtPParagraphSplitter instance.

    Args:
        wtp_paragraph_splitter (Optional[WtPParagraphSplitter]): An optional
            instance of WtPParagraphSplitter to share the model.

    Returns:
        WtPSentenceSplitter: The initialized sentence splitter instance.
    """
    global wtp_sentence_splitter
    if wtp_sentence_splitter is None:
        wtp_paragraph_splitter = wtp_paragraph_splitter or await get_wtp_paragraph_splitter()
        # Share the splitter from WtPParagraphSplitter to avoid redundant model loading
        wtp_sentence_splitter = WtPSentenceSplitter(
            config={"model": settings.WtP_model,
                    "use_cuda": settings.WtP_cuda},
            splitter=wtp_paragraph_splitter.splitter)
    return wtp_sentence_splitter


async def chunk(content: Union[str, bytes], file_extension: str = "txt",
                chunking_strategy: str = "paragraphs",
                filter_sents: bool = False) -> List[str]:
    """
    Helper to split a document into smaller chunks for ingestion.

    Args:
        content (Union[str, bytes]): The content to be chunked. Can be a string or bytes.
        file_extension (str): The file extension of the content (e.g., "txt", "pdf", "html").
        chunking_strategy (str): The strategy to use for chunking ("paragraphs" or "sentences").
        filter_sents (bool): If True, filters out sentences that do not contain a newline character.

    Returns:
        List[str]: A list of strings, where each string is a chunk of the original content.
    """
    # Determine chunking strategy and splitter
    splitter = None

    if file_extension == "pdf":
        splitter = PDFParagraphSplitter()
    elif file_extension == "doc":
        splitter = DOCParagraphSplitter()
    elif file_extension == "docx":
        splitter = DOCxParagraphSplitter()
    elif file_extension == "html":
        splitter = HTMLParagraphSplitter()
    elif file_extension == "md":
        splitter = MarkdownParagraphSplitter()
    elif chunking_strategy == "sentences":
        splitter = await get_wtp_sentence_splitter()
    elif chunking_strategy == "paragraphs":
        splitter = await get_wtp_paragraph_splitter()

    if splitter:
        if file_extension in ["pdf", "doc", "docx"]:
            # These chunkers require a file path
            with NamedTemporaryFile("wb") as temp_file:
                if isinstance(content, str):
                    temp_file.write(content.encode("utf-8"))
                else:
                    temp_file.write(content)
                temp_file.flush()
                chunked_data = list(splitter.chunk(temp_file.name)) # iterate now before tmp file is deleted
                if filter_sents:
                    return [p for p in chunked_data if "\n" in p]
                else:
                    return chunked_data
        else:
            # Text-based chunkers
            if isinstance(content, str):
                return splitter.chunk(content)
            else:
                return splitter.chunk(content.decode("utf-8"))
    else:
        # No specific splitter, treat as one chunk
        if isinstance(content, str):
            return [content]
        else:
            return [content.decode("utf-8")]


async def get_reranker_plugin() -> MultipleChoiceSolver:
    """
    FastAPI dependency that provides the initialized TldrSolver plugin instance.
    Loads the plugin based on settings.

    Returns:
        MultipleChoiceSolver: The initialized MultipleChoiceSolver plugin.

    Raises:
        HTTPException: If the plugin fails to load.
    """
    global reranker_plugin
    if reranker_plugin is None:
        try:
            clazz = load_multiple_choice_solver_plugin(settings.reranker_plugin)
            if not clazz:
                raise FileNotFoundError(f"Failed to find MultipleChoiceSolver plugin: {settings.reranker_plugin}")
            reranker_plugin = clazz(settings.reranker_config)
            LOG.info(f"Loaded MultipleChoiceSolver plugin: {settings.reranker_plugin}")
        except Exception as e:
            LOG.error(f"Error loading MultipleChoiceSolver plugin: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Failed to load TldrSolver plugin: {e}")
    return reranker_plugin


async def get_summarizer_plugin() -> TldrSolver:
    """
    FastAPI dependency that provides the initialized TldrSolver plugin instance.
    Loads the plugin based on settings.

    Returns:
        TldrSolver: The initialized TldrSolver plugin.

    Raises:
        HTTPException: If the plugin fails to load.
    """
    global summarizer_plugin
    if summarizer_plugin is None:
        try:
            clazz = load_tldr_solver_plugin(settings.summarizer_plugin)
            if not clazz:
                raise FileNotFoundError(f"Failed to find TldrSolver plugin: {settings.summarizer_plugin}")
            summarizer_plugin = clazz(settings.summarizer_config)
            LOG.info(f"Loaded TldrSolver plugin: {settings.summarizer_plugin}")
        except Exception as e:
            LOG.error(f"Error loading TldrSolver plugin: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Failed to load TldrSolver plugin: {e}")
    return summarizer_plugin


async def get_default_persona() -> Persona:
    """
    Asynchronously loads and returns the default persona.

    This function acts as a dependency for FastAPI endpoints, ensuring
    the persona is loaded once and reused across requests.
    It raises HTTPException if the persona file is not configured,
    not found, or invalid.

    Returns:
        Persona: The loaded OVOS Persona instance.

    Raises:
        HTTPException: If persona configuration or loading fails.
    """
    global default_persona
    if default_persona is None:
        persona_data: dict = settings.persona_config
        try:
            default_persona = Persona(persona_data["name"], persona_data)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail=f"JSONDecodeError for persona config: {persona_data}") from e
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail=f"Invalid JSON config for persona: {persona_data}") from e
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Failed to load persona: {e}") from e
    return default_persona


@persona_router.get("/status", response_model=Status)
async def stats(persona: Persona = Depends(get_default_persona),
                summarizer: TldrSolver = Depends(get_summarizer_plugin),
                reranker: MultipleChoiceSolver = Depends(get_reranker_plugin)) -> Status:
    """
    Returns the status of the currently loaded persona.

    Args:
        persona (Persona): The persona instance, injected by FastAPI's dependency.

    Returns:
        Status: An object containing the persona's name, loaded solvers, and models.
    """
    version = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_MINOR}a{VERSION_ALPHA}" \
        if VERSION_ALPHA else f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_MINOR}"
    models = {s: persona.config.get(s, {}).get("model")
              for s in persona.solvers.loaded_modules.keys()}
    if reranker.config.get("model"):
        models[settings.reranker_plugin] = reranker.config["model"]
    if summarizer.config.get("model"):
        models[settings.summarizer_plugin] = summarizer.config["model"]
    return Status(
        persona=persona.name,
        solvers=list(persona.solvers.loaded_modules.keys()),
        summarizer_plugin=settings.summarizer_plugin,
        reranker_plugin=settings.reranker_plugin,
        models=models,
        version=version
    )


@persona_router.post("/summarize")
async def summarize(request: SummarizeRequest = Body(...),
                    summarizer: TldrSolver = Depends(get_summarizer_plugin)) -> Response:
    """
    Summarizes the input text using the configured summarizer plugin.

    Args:
        request (SummarizeRequest): The request body containing the text to summarize.
        summarizer (TldrSolver): The summarizer plugin instance, injected by FastAPI.

    Returns:
        Response: A FastAPI Response object with the summarized content as plain text.

    Raises:
        HTTPException: If the summarization request fails or the plugin encounters an error.
    """
    summarized_content: str = summarizer.get_tldr(request.input)
    return Response(content=summarized_content, media_type="text/markdown")


@persona_router.post("/rerank")
async def rerank(request: RerankRequest = Body(...),
                 reranker: MultipleChoiceSolver = Depends(get_reranker_plugin)) -> List[Tuple[float, str]]:
    """
    Reranks a list of documents based on a given query using the configured reranker plugin.

    Args:
        request (RerankRequest): The request body containing the query and documents to rerank.
        reranker (MultipleChoiceSolver): The reranker plugin instance, injected by FastAPI.

    Returns:
        List[Tuple[float, str]]: A list of tuples, where each tuple contains a relevance score (float)
                                  and the corresponding document (str), sorted by relevance.

    Raises:
        HTTPException: If the reranking request fails or the plugin encounters an error.
    """
    return reranker.rerank(request.input, request.documents)


@persona_router.post("/segment")
async def segment(request: SegmentRequest = Body(...)) -> List[str]:
    """
    Segments the input text or content from a URL into smaller chunks.

    Args:
        request (SegmentRequest): The request body containing the input text or URL.

    Returns:
        List[str]: A list of strings, where each string is a segment of the input.

    Raises:
        HTTPException: If the segmentation request fails (e.g., invalid URL or chunking error).
    """
    if request.input.startswith("http"):
        try:
            response = requests.get(request.input)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            return await chunk(response.text, "html")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Failed to fetch content from URL: {e}") from e
    return await chunk(request.input)


@persona_router.post("/segment/file")
async def segment_file(file: UploadFile = File(..., description="The file to upload.")) -> List[str]:
    """
    Segments an uploaded file into smaller chunks based on its file extension.

    Args:
        file (UploadFile): The uploaded file to be segmented.

    Returns:
        List[str]: A list of strings, where each string is a segment of the file content.

    Raises:
        HTTPException: If the file processing fails (e.g., unable to read file, unsupported extension).
    """
    content_bytes: bytes = await file.read()
    extension: str = file.filename.split(".")[-1] if "." in file.filename else "txt"
    return await chunk(content_bytes, extension)
