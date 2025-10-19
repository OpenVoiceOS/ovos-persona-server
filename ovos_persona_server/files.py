"""
FastAPI router for OpenAI Files API endpoints.

This module provides an implementation of OpenAI's Files API,
allowing for file uploads, retrieval, listing, and deletion.
It now uses SQLAlchemy ORM for database interactions.
"""

import hashlib
import os
import random
import string
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, List

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Query, Depends
from fastapi import FastAPI
from fastapi.responses import Response
from ovos_utils.log import LOG
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ovos_persona_server.config import settings
from ovos_persona_server.metadata import get_async_db, File as FileORM
from ovos_persona_server.schemas.openai_files import FileObject, FileListResponse, FilePurpose, DeleteFileResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Provide a minimal lifespan context for the FastAPI router; performs no startup or shutdown actions.
    """
    yield
    # No specific shutdown logic needed for these dependencies currently


files_router = APIRouter(prefix="/v1/files", tags=["files"], lifespan=lifespan)


# Dependency to get a database session
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide an asynchronous SQLAlchemy session for use as a FastAPI dependency.
    
    Returns:
        AsyncSession: An asynchronous SQLAlchemy session yielded to the dependency consumer.
    """
    async for session in get_async_db():
        yield session


def _generate_id(prefix: str = "") -> str:
    """
    Generate a short, unique identifier with an optional prefix.
    
    Parameters:
        prefix (str): Optional string to prepend to the identifier (e.g., "file_").
    
    Returns:
        str: The prefix followed by 24 random alphanumeric characters.
    """
    return f"{prefix}{''.join(random.choices(string.ascii_letters + string.digits, k=24))}"


def _get_current_timestamp() -> int:
    """
    Get the current Unix timestamp.
    
    Returns:
        int: Current Unix timestamp in seconds since the Unix epoch.
    """
    return int(time.time())


def _get_file_storage_dir() -> str:
    """
    Ensure the configured file storage directory exists and return its absolute path.
    
    The directory will be created if it does not already exist.
    
    Returns:
        str: Absolute path to the file storage directory.
    """
    storage_dir = os.path.abspath(settings.file_storage_path)
    os.makedirs(storage_dir, exist_ok=True)
    return storage_dir


# --- Files Endpoints ---

@files_router.post("/", response_model=FileObject, status_code=status.HTTP_200_OK)
async def upload_file(
        file: UploadFile = File(..., description="The file to upload."),
        purpose: FilePurpose = Form(...,
                                    description="The intended purpose of the file. Currently only 'assistants' is supported."),
        db: AsyncSession = Depends(get_db)
) -> FileObject:
    """
        Upload a file, store its content and metadata according to the configured storage strategy, and return the stored FileObject.
        
        The endpoint detects duplicates by computing a SHA-256 hash of the file contents combined with the provided purpose; if a matching file already exists it returns the existing FileObject instead of creating a new record.
        
        Parameters:
            file (UploadFile): The uploaded file.
            purpose (FilePurpose): Intended purpose of the file; currently only the `assistants` purpose is supported and is stored as its enum value.
        
        Returns:
            FileObject: The created or existing file metadata and identifiers.
        
        Raises:
            HTTPException: If file processing, storage, or database persistence fails.
        """
    content_bytes: bytes = await file.read()
    file_size: int = len(content_bytes)
    content_hash: str = hashlib.sha256(content_bytes).hexdigest() # Calculate SHA256 hash

    # Check for existing file with the same content hash and purpose
    existing_file_query = select(FileORM).where(
        FileORM.content_hash == content_hash,
        FileORM.purpose == purpose.value
    )
    existing_file_result = await db.execute(existing_file_query)
    existing_file_orm = existing_file_result.scalars().first()

    if existing_file_orm:
        # If a duplicate is found, return the existing file object
        LOG.debug(f"Duplicate file detected for purpose '{purpose.value}' with hash {content_hash}. "
                  f"Returning existing file {existing_file_orm.id}.")
        return FileObject(**existing_file_orm.to_dict())

    # No duplicate found, proceed with new file creation
    file_id: str = _generate_id("file_")
    current_time: int = _get_current_timestamp()
    local_path: Optional[str] = None

    try:
        # Handle disk storage
        if settings.file_storage_strategy in ["disk", "both"]:
            storage_dir: str = _get_file_storage_dir()
            unique_filename: str = f"{file_id}_{file.filename}"
            local_path = os.path.join(storage_dir, unique_filename)
            with open(local_path, "wb") as f:
                f.write(content_bytes)

        # Create ORM object and add to session
        new_file_orm = FileORM(
            id=file_id,
            object="file",
            bytes=file_size,
            created_at=current_time,
            filename=file.filename,
            purpose=purpose.value,  # Store enum value as string
            status="uploaded",
            status_details=None,
            local_path=local_path, # Store local path if saved to disk
            content_data=content_bytes if settings.file_storage_strategy in ["database", "both"] else None,
            content_hash=content_hash # Store the calculated hash
        )
        db.add(new_file_orm)
        await db.commit()
        await db.refresh(new_file_orm)  # Refresh to get any default values set by DB

        # Convert ORM object to Pydantic model for response
        return FileObject(**new_file_orm.to_dict())

    except Exception as e:
        # Clean up the file from disk if DB insertion fails
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to upload file: {e}") from e


@files_router.get("/", response_model=FileListResponse, status_code=status.HTTP_200_OK)
async def list_files(
        purpose: Optional[FilePurpose] = Query(None, description="Only return files with the given purpose."),
        limit: int = Query(20, ge=1, le=100),
        order: str = Query("desc", pattern="^(asc|desc)$"),
        after: Optional[str] = None,
        before: Optional[str] = None,
        db: AsyncSession = Depends(get_db)
) -> FileListResponse:
    """
        List files visible to the caller, optionally filtered by purpose, sorted, and cursor-paginated.
        
        Pagination is applied in memory: `after` and `before` are file ID cursors used to slice the sorted result set, and `limit` bounds the number of returned items (1–100). `order` controls sorting by creation time and accepts "asc" or "desc".
        
        Parameters:
            after (Optional[str]): ID of the file to start after (exclusive) when paginating.
            before (Optional[str]): ID of the file to end before (exclusive) when paginating.
            order (str): Sort order for results; either "asc" or "desc".
        
        Returns:
            FileListResponse: Contains the list of FileObject items in `data`, plus `first_id`, `last_id`, and `has_more` pagination metadata.
        """
    query = select(FileORM)
    if purpose:
        query = query.where(FileORM.purpose == purpose.value)

    # Fetch all relevant files first to handle pagination logic in Python
    # For large datasets, this should be optimized with database-level pagination
    result = await db.execute(query)
    all_files_orm = result.scalars().all()

    # Convert ORM objects to Pydantic models and sort
    all_files: List[FileObject] = [FileObject(**f.to_dict()) for f in all_files_orm]
    all_files.sort(key=lambda x: x.created_at, reverse=(order == "desc"))

    start_index: int = 0
    end_index: int = len(all_files)

    if after:
        try:
            after_index: int = next(i for i, file in enumerate(all_files) if file.id == after)
            start_index = after_index + 1
        except StopIteration:
            pass

    if before:
        try:
            before_index: int = next(i for i, file in enumerate(all_files) if file.id == before)
            end_index = before_index
        except StopIteration:
            pass

    paginated_files: List[FileObject] = all_files[start_index:end_index]
    data: List[FileObject] = paginated_files[:limit]
    has_more: bool = len(paginated_files) > limit

    first_id: Optional[str] = data[0].id if data else None
    last_id: Optional[str] = data[-1].id if data else None

    return FileListResponse(
        object="list",
        data=data,
        first_id=first_id,
        last_id=last_id,
        has_more=has_more
    )


@files_router.get("/{file_id}", response_model=FileObject, status_code=status.HTTP_200_OK)
async def retrieve_file(file_id: str, db: AsyncSession = Depends(get_db)) -> FileObject:
    """
    Retrieve metadata for a file by its ID.
    
    Parameters:
        file_id (str): The ID of the file to retrieve.
    
    Returns:
        FileObject: The file's metadata and attributes.
    
    Raises:
        HTTPException: If no file with the given ID exists (404).
    """
    result = await db.execute(select(FileORM).where(FileORM.id == file_id))
    file_orm = result.scalars().first()
    if not file_orm:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    return FileObject(**file_orm.to_dict())


@files_router.delete("/{file_id}", response_model=DeleteFileResponse, status_code=status.HTTP_200_OK)
async def delete_file(file_id: str, db: AsyncSession = Depends(get_db)) -> DeleteFileResponse:
    """
    Delete a file by its ID and remove any stored content.
    
    If the file is stored on disk and the configured storage strategy allows it, the on-disk file is removed after the database record is deleted.
    
    Returns:
        DeleteFileResponse: Confirmation containing the deleted file ID, object type 'file.deleted', and deleted=True.
    
    Raises:
        HTTPException: If the file is not found (status 404) or if deleting the on-disk file fails (status 500).
    """
    # First, retrieve the file to get its local_path and confirm existence
    result = await db.execute(select(FileORM).where(FileORM.id == file_id))
    file_orm = result.scalars().first()
    if not file_orm:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    local_path = file_orm.local_path

    # Then, delete the record from the database (this handles content_data if stored in DB)
    await db.delete(file_orm)
    await db.commit()

    # Finally, delete the file from disk if it exists and strategy allows
    if settings.file_storage_strategy in ["disk", "both"] and local_path and os.path.exists(local_path):
        try:
            os.remove(local_path)
        except OSError as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Error deleting file from disk {local_path}: {e}") from e

    return DeleteFileResponse(id=file_id, object="file.deleted", deleted=True)


@files_router.get("/{file_id}/content", status_code=status.HTTP_200_OK)
async def retrieve_file_content(file_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    """
    Retrieve the raw content of a stored file and return it with an appropriate media type.
    
    Parameters:
        file_id (str): Identifier of the file to fetch.
    
    Returns:
        response (Response): An HTTP response whose body is the file's raw bytes and whose media type is inferred from the filename (defaults to "application/octet-stream").
    
    Raises:
        HTTPException: If the file does not exist (404), if the content is not available under the current storage strategy (404), or if reading the content fails (500).
    """
    result = await db.execute(
        select(FileORM.local_path, FileORM.filename, FileORM.purpose, FileORM.content_data).where(FileORM.id == file_id))
    row = result.first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    local_path, filename, purpose, content_from_db = row.local_path, row.filename, row.purpose, row.content_data
    content: Optional[bytes] = None

    # Prioritize disk storage if available and strategy allows
    if settings.file_storage_strategy in ["disk", "both"] and local_path and os.path.exists(local_path):
        try:
            with open(local_path, "rb") as f:
                content = f.read()
        except IOError as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Failed to read file content from disk: {e}") from e
    elif settings.file_storage_strategy in ["database", "both"] and content_from_db is not None:
        content = content_from_db
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="File content not found or not accessible with current storage strategy.")

    if content is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="File content could not be retrieved.")

    # Determine media type based on filename (simple heuristic)
    media_type: str = "application/octet-stream"
    if filename:
        if filename.endswith(".json"):
            media_type = "application/json"
        elif filename.endswith(".txt"):
            media_type = "text/plain"
        elif filename.endswith(".csv"):
            media_type = "text/csv"
        elif filename.endswith((".png", ".jpg", ".jpeg", ".gif")):
            media_type = f"image/{filename.split('.')[-1]}"
        elif filename.endswith(".pdf"):
            media_type = "application/pdf"
        # Add more media types as needed

    return Response(content=content, media_type=media_type)