"""
Pydantic schemas for the OVOS Persona Server API.

This module defines the data models for requests and responses across
various API endpoints, including OpenAI-compatible chat and completions,
embeddings, Ollama-specific interactions, and OpenAI Vector Stores.
"""
from enum import Enum
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field


class SegmentationStrategy(str, Enum):
    """
    Enumerates the available text segmentation strategies.
    """
    SENTENCES = "sentences"
    PARAGRAPHS = "paragraphs"


class Status(BaseModel):
    """
    Represents the current status and configuration of the OVOS Persona Server.

    Attributes:
        persona (str): The name or identifier of the active persona.
        summarizer_plugin (str): The name of the currently loaded summarizer plugin.
        reranker_plugin (str): The name of the currently loaded reranker plugin.
        solvers (List[str]): A list of identifiers for all loaded solver plugins.
        models (Dict[str, str]): A dictionary mapping solver names to their configured model identifiers or versions.
        version (str): The semantic version string of the Persona Server.
    """
    persona: str = Field(..., description="The name or identifier of the current persona being used.")
    summarizer_plugin: str = Field(..., description="The name of the summarizer plugin currently in use.")
    reranker_plugin: str = Field(..., description="The name of the reranker plugin currently in use.")
    solvers: List[str] = Field(..., description="A list of available solvers that the server can utilize.")
    models: Dict[str, str] = Field(..., description="A dictionary mapping model names to their identifiers or versions.")
    version: str = Field(..., description="The version of the Persona Server.")


class SummarizeRequest(BaseModel):
    """
    Represents a request to summarize a given text.

    Attributes:
        input (str): The text content that needs to be summarized.
    """
    input: str = Field(..., description="Text to be summarized.")


class SegmentRequest(BaseModel):
    """
    Represents a request for segmenting text or content from a URL.

    Attributes:
        input (str): The text content or a URL pointing to content that needs to be segmented.
                     If a URL is provided, the content at that URL will be fetched and analyzed.
        strategy (Optional[SegmentationStrategy]): The strategy to use for segmentation.
                                                   Can be 'paragraphs' or 'sentences'.
                                                   This attribute is ignored if the input is a URL,
                                                   as URL content is typically segmented by paragraphs.
    """
    input: str = Field("https://openvoiceos.org", description="Text content or URL to be processed.")
    strategy: Optional[SegmentationStrategy] = Field(
        SegmentationStrategy.PARAGRAPHS,
        description="Segmentation strategy: 'paragraphs' or 'sentences'. Ignored if input is a URL."
    )


class RerankRequest(BaseModel):
    """
    Represents a request to rerank a list of documents based on a given input query.

    Attributes:
        input (str): The input text or query string that serves as the basis for reranking the documents.
        documents (List[str]): A list of document strings or content that needs to be reranked
                               against the provided input.
    """
    input: str = Field(..., description="The input text or query for reranking.")
    documents: List[str] = Field(..., description="A list of documents (strings) to be reranked.")