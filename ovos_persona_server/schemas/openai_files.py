"""
Pydantic models for OpenAI-compatible Files API endpoints.

This module defines the data structures for file objects and lists,
aligning with the OpenAI API specification for file management. These models
are used to validate request and response bodies for file-related operations.
"""

from enum import Enum
from typing import List, Optional, Literal, Any

from pydantic import BaseModel, Field, ConfigDict


class FilePurpose(str, Enum):
    """
    The intended purpose of the uploaded file.
    """
    ASSISTANTS = "assistants"
    ASSISTANTS_OUTPUT = "assistants_output"
    BATCH = "batch"
    BATCH_OUTPUT = "batch_output"
    FINE_TUNE = "fine-tune"
    FINE_TUNE_RESULTS = "fine-tune-results"
    VISION = "vision"
    USER_DATA = "user_data"
    EVALS = "evals"


class FileObject(BaseModel):
    """
    Represents a document that has been uploaded to OpenAI.
    """
    id: str = Field(..., description="The file identifier, which can be referenced in the API endpoints.")
    object: Literal["file"] = Field("file", description="The object type, which is always `file`.")
    bytes: int = Field(..., description="The size of the file, in bytes.")
    created_at: int = Field(..., description="The Unix timestamp (in seconds) for when the file was created.")
    filename: str = Field(..., description="The name of the file.")
    purpose: FilePurpose = Field(..., description="The intended purpose of the file.")
    status: Optional[str] = Field("uploaded", description="Deprecated. The current status of the file, which can be either `uploaded`, `processed`, or `error`.")
    status_details: Optional[str] = Field(None, description="Deprecated. For details on why a fine-tuning training file failed validation, see the `error` field on `fine_tuning.job`.")


class FileListResponse(BaseModel):
    """
    A list of File objects.
    """
    object: Literal["list"] = Field("list", description="The object type, which is always `list`.")
    data: List[FileObject] = Field(..., description="A list of file objects.")
    has_more: bool = Field(False, description="Indicates if there are more files available.")
    first_id: Optional[str] = Field(None, description="The ID of the first object in the list.")
    last_id: Optional[str] = Field(None, description="The ID of the last object in the list.")


class DeleteFileResponse(BaseModel):
    """
    Confirmation of a file deletion operation.
    """
    id: str = Field(..., description="The ID of the deleted file.")
    object: Literal["file.deleted"] = Field("file.deleted", description="The object type, which is always `file.deleted`.")
    deleted: bool = Field(..., description="A flag indicating if the file was successfully deleted.")


class CreateFileRequest(BaseModel):
    """
    Request body for uploading a file.
    """
    # Note: FastAPI handles UploadFile separately, Pydantic model for request body
    # typically doesn't directly validate io.IOBase unless it's a custom validator.
    # This model is mostly for documentation/schema generation.
    file: Any = Field(..., description="The File object (not file name) to be uploaded. This will be handled as UploadFile in FastAPI.")
    purpose: FilePurpose = Field(..., description="The intended purpose of the uploaded file.")

    model_config = ConfigDict(arbitrary_types_allowed=True) # Allow 'file: Any' to pass Pydantic validation



