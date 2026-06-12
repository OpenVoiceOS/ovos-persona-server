"""
Pydantic models for OpenAI-compatible Vector Stores API endpoints.

This module defines the data structures for creating, managing, and interacting
with vector stores, their associated files, and file batches, ensuring full
compatibility with the OpenAI API specification.
"""

from enum import Enum
from typing import List, Optional, Dict, Union, Any, Literal, Annotated

from pydantic import BaseModel, Field, ConfigDict, RootModel

# For easy typing of metadata, which is a dictionary with string keys and string values
Metadata = Dict[str, str]


# --- Shared Objects and Enums ---

class VectorStoreFileStatus(str, Enum):
    """Status of a vector store file."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class VectorStoreStatus(str, Enum):
    """Status of a vector store."""
    EXPIRED = "expired"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class VectorStoreFileCounts(BaseModel):
    """Counts of files in a vector store by their status."""
    in_progress: int = Field(..., description="The number of files that are currently being processed.")
    completed: int = Field(..., description="The number of files that have been successfully processed.")
    failed: int = Field(..., description="The number of files that have failed to process.")
    cancelled: int = Field(..., description="The number of files that were cancelled.")
    total: int = Field(..., description="The total number of files.")


class VectorStoreExpirationAfter(BaseModel):
    """The expiration policy for a vector store."""
    anchor: Literal["last_active_at"] = Field(..., description="Time anchor for the expiration time. Currently only 'last_active_at' is supported.")
    minutes: Annotated[int, Field(ge=1)] = Field(..., description="The number of minutes after the `anchor` time that the vector store will expire.")


class VectorStoreFileLastError(BaseModel):
    """The last error associated with a vector store file."""
    code: str = Field(..., description="The error code for the last error.")
    message: str = Field(..., description="The message for the last error.")


class VectorStoreFileChunkingStrategyStatic(BaseModel):
    """
    The static chunking strategy which uses a fixed chunk size.
    """
    type: Literal["static"] = Field("static", description="The type of chunking strategy.")
    static: Optional[Dict[str, Any]] = Field({}, description="Details about the static chunking strategy. This object is currently empty.")


class VectorStoreFileChunkingStrategyAuto(BaseModel):
    """
    The auto chunking strategy which automatically determines the best chunk size.
    """
    type: Literal["auto"] = Field("auto", description="The type of chunking strategy.")
    auto: Optional[Dict[str, Any]] = Field({}, description="Details about the auto chunking strategy. This object is currently empty.")


class VectorStoreFileChunkingStrategyOther(BaseModel):
    """
    The other chunking strategy.
    """
    type: Literal["other"] = Field("other", description="The type of chunking strategy.")
    other: Optional[Dict[str, Any]] = Field({}, description="Details about the other chunking strategy. This object is currently empty.")


class VectorStoreFileChunkingStrategy(RootModel[Union[
    VectorStoreFileChunkingStrategyStatic,
    VectorStoreFileChunkingStrategyAuto,
    VectorStoreFileChunkingStrategyOther
]]):
    """
    The chunking strategy used to chunk the file(s).
    """
    pass


class VectorStoreObject(BaseModel):
    """
    Represents a vector store.
    """
    id: str = Field(..., description="The vector store identifier, which can be referenced in the API endpoints.")
    object: Literal["vector_store"] = Field("vector_store", description="The object type, which is always `vector_store`.")
    created_at: int = Field(..., description="The Unix timestamp (in seconds) for when the vector store was created.")
    name: str = Field(..., description="The name of the vector store.")
    usage_bytes: int = Field(..., description="The total number of bytes used by the files in the vector store.")
    file_counts: VectorStoreFileCounts = Field(..., description="The number of files in the vector store and their statuses.")
    status: VectorStoreStatus = Field(..., description="The status of the vector store.")
    expires_after: Optional[VectorStoreExpirationAfter] = Field(None, description="The expiration policy for the vector store.")
    expires_at: Optional[int] = Field(None, description="The Unix timestamp (in seconds) for when the vector store will expire.")
    last_active_at: Optional[int] = Field(None, description="The Unix timestamp (in seconds) for when the vector store was last active.")
    metadata: Optional[Metadata] = Field(None, description="Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard. Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.")


class VectorStoreDeleted(BaseModel):
    """
    Confirmation of a vector store deletion operation.
    """
    id: str = Field(..., description="The ID of the deleted vector store.")
    object: Literal["vector_store.deleted"] = Field("vector_store.deleted", description="The object type, which is always `vector_store.deleted`.")
    deleted: bool = Field(..., description="A flag indicating if the vector store was successfully deleted.")


class VectorStoreFileObject(BaseModel):
    """
    Represents a vector store file.
    """
    id: str = Field(..., description="The file identifier, which can be referenced in the API endpoints.")
    object: Literal["vector_store.file"] = Field("vector_store.file", description="The object type, which is always `vector_store.file`.")
    usage_bytes: int = Field(..., description="The total number of bytes used by the file.")
    created_at: int = Field(..., description="The Unix timestamp (in seconds) for when the vector store file was created.")
    vector_store_id: str = Field(..., description="The ID of the vector store that the file belongs to.")
    status: VectorStoreFileStatus = Field(..., description="The status of the vector store file.")
    last_error: Optional[VectorStoreFileLastError] = Field(None, description="The last error associated with this vector store file.")
    chunking_strategy: Optional[VectorStoreFileChunkingStrategy] = Field(None, description="The chunking strategy used to chunk the file(s).")
    attributes: Optional[Dict[str, Any]] = Field(None, description="A set of key-value pairs that can be attached to a vector store file. This can be useful for storing additional information about the vector store file in a structured format. Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.")


class VectorStoreFileDeleted(BaseModel):
    """
    Confirmation of a vector store file deletion operation.
    """
    id: str = Field(..., description="The ID of the deleted vector store file.")
    object: Literal["vector_store.file.deleted"] = Field("vector_store.file.deleted", description="The object type, which is always `vector_store.file.deleted`.")
    deleted: bool = Field(..., description="A flag indicating if the vector store file was successfully deleted.")


class CreateVectorStoreRequest(BaseModel):
    """
    Request body for creating a vector store.
    """
    file_ids: Optional[List[str]] = Field(None, description="A list of [file](/docs/api-reference/files/object) IDs that the vector store should use. Useful for tools like file_search that can access files stored in the vector store.")
    name: Optional[str] = Field(None, description="The name of the vector store.")
    expires_after: Optional[VectorStoreExpirationAfter] = Field(None, description="The expiration policy for the vector store.")
    metadata: Optional[Metadata] = Field(None, description="Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard. Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.")


class ListVectorStoresResponse(BaseModel):
    """
    A list of vector stores.
    """
    object: Literal["list"] = Field("list", description="The object type, which is always `list`.")
    data: List[VectorStoreObject] = Field(..., description="A list of vector store objects.")
    first_id: Optional[str] = Field(None, description="The ID of the first object in the list.")
    last_id: Optional[str] = Field(None, description="The ID of the last object in the list.")
    has_more: bool = Field(..., description="A flag indicating whether there are more objects to retrieve.")


class UpdateVectorStoreRequest(BaseModel):
    """
    Request body for updating a vector store.
    """
    name: Optional[str] = Field(None, description="The name of the vector store.")
    expires_after: Optional[VectorStoreExpirationAfter] = Field(None, description="The expiration policy for the vector store.")
    metadata: Optional[Metadata] = Field(None, description="Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard. Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.")


class CreateVectorStoreFileRequest(BaseModel):
    """
    Request body for adding a file to a vector store.
    """
    file_id: str = Field(..., description="A [file](/docs/api-reference/files/object) ID that the vector store should use. Useful for tools like `file_search` that can access files stored in the vector store.")
    chunking_strategy: Optional[VectorStoreFileChunkingStrategy] = Field(default_factory=VectorStoreFileChunkingStrategyAuto, description="The chunking strategy used to chunk the file(s). If not provided, will use the default strategy configured for the model or vector store.")
    attributes: Optional[Dict[str, Any]] = Field({}, description="A set of key-value pairs that can be attached to a vector store file. This can be useful for storing additional information about the vector store file in a structured format. Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.")


class ListVectorStoreFilesResponse(BaseModel):
    """
    A list of vector store files.
    """
    object: Literal["list"] = Field("list", description="The object type, which is always `list`.")
    data: List[VectorStoreFileObject] = Field(..., description="A list of vector store file objects.")
    first_id: Optional[str] = Field(None, description="The ID of the first object in the list.")
    last_id: Optional[str] = Field(None, description="The ID of the last object in the list.")
    has_more: bool = Field(..., description="A flag indicating whether there are more objects to retrieve.")


class UpdateVectorStoreFileAttributesRequest(BaseModel):
    """
    Request body for updating attributes on a vector store file.
    """
    attributes: Optional[Dict[str, Any]] = Field(None, description="A set of key-value pairs that can be attached to a vector store file. This can be useful for storing additional information about the vector store file in a structured format. Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.")


class ComparisonFilter(BaseModel):
    """
    A filter that compares a file attribute to a value.
    """
    field: str = Field(..., description="The name of the attribute to filter by.")
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"] = Field(..., description="The comparison operator.")
    value: Union[str, int, float, bool, List[Union[str, int, float, bool]]] = Field(..., description="The value to compare against.")


class CompoundFilter(BaseModel):
    """
    A filter that combines multiple filters with a logical operator.
    """
    operator: Literal["and", "or"] = Field(..., description="The logical operator to combine filters.")
    filters: Annotated[List[Union[ComparisonFilter, 'CompoundFilter']], Field(min_length=1)] = Field(..., description="Array of filters to combine.")

    model_config = ConfigDict(arbitrary_types_allowed=True) # Needed for self-referencing types


class RankingOptions(BaseModel):
    """Ranking options for the search."""
    ranker: Optional[Literal["auto", "default_2024_08_21"]] = Field("auto", description="The ranker to use for the file search.")
    score_threshold: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(0.0, description="The score threshold for the file search.")


class SearchVectorStoreFileRequest(BaseModel):
    """Request body for searching a vector store."""
    query: str = Field(..., description="A query string for a search.")
    max_num_results: Optional[Annotated[int, Field(ge=1, le=50)]] = Field(20, description="The maximum number of results to return.")
    filters: Optional[Union[ComparisonFilter, CompoundFilter]] = Field(None, description="A filter to apply based on file attributes.")
    ranking_options: Optional[RankingOptions] = Field(None, description="Ranking options for search.")
    rewrite_query: Optional[bool] = Field(False, description="Whether to rewrite the natural language query for vector search.")


class SearchResultChunk(BaseModel):
    """Represents a single search result chunk from a vector store search."""
    type: Literal["file_search"] = Field("file_search")
    file_id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class VectorStoreSearchResponse(BaseModel):
    """Response for a vector store search."""
    object: Literal["list"] = Field("list")
    data: List[SearchResultChunk]
    has_more: bool = Field(False)

# No need for CompoundFilter.update_forward_refs() with Pydantic V2 and RootModel
# It's handled automatically if `from __future__ import annotations` is used,
# or by `model_config = ConfigDict(arbitrary_types_allowed=True)` for self-referencing types.
