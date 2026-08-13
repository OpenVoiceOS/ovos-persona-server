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
    Manages the lifespan of the FastAPI application.
    Database initialization is now handled globally by init_db in database.py.
    """
    yield
    # No specific shutdown logic needed for these dependencies currently


files_router = APIRouter(prefix="/openai/v1/files", tags=["files"])


# Dependency to get a database session
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides an asynchronous SQLAlchemy database session.
    """
    async for session in get_async_db():
        yield session


def _generate_id(prefix: str = "") -> str:
    """
    Generates a unique ID for API objects.

    Args:
        prefix (str): A prefix for the ID (e.g., "file_").

    Returns:
        str: A unique ID string.
    """
    return f"{prefix}{''.join(random.choices(string.ascii_letters + string.digits, k=24))}"


def _get_current_timestamp() -> int:
    """
    Returns the current Unix timestamp.

    Returns:
        int: The current Unix timestamp.
    """
    return int(time.time())


def _get_file_storage_dir() -> str:
    """
    Ensures the file storage directory exists and returns its absolute path.
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
    Upload a file that can be used across various API endpoints.
    Checks for duplicate file content based on hash and purpose.

    Args:
        file (UploadFile): The file to upload.
        purpose (FilePurpose): The intended purpose of the file (e.g., 'assistants').
        db (AsyncSession): The SQLAlchemy asynchronous session, injected via dependency.

    Returns:
        FileObject: The newly created or existing file object if a duplicate is found.

    Raises:
        HTTPException: If the purpose is invalid or file processing fails.
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
    Returns a list of files that belong to the user's organization.

    Args:
        purpose (Optional[FilePurpose]): Filter files by their purpose.
        limit (int): A limit on the number of objects to be returned.
        order (str): Sort order for the results. Can be 'asc' or 'desc'.
        after (Optional[str]): A cursor for use in pagination.
        before (Optional[str]): A cursor for use in pagination.
        db (AsyncSession): The SQLAlchemy asynchronous session, injected via dependency.

    Returns:
        FileListResponse: A list of file objects.
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
    Returns information about a specific file.

    Args:
        file_id (str): The ID of the file to retrieve.
        db (AsyncSession): The SQLAlchemy asynchronous session, injected via dependency.

    Returns:
        FileObject: The retrieved file object.

    Raises:
        HTTPException: If the file is not found.
    """
    result = await db.execute(select(FileORM).where(FileORM.id == file_id))
    file_orm = result.scalars().first()
    if not file_orm:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    return FileObject(**file_orm.to_dict())


@files_router.delete("/{file_id}", response_model=DeleteFileResponse, status_code=status.HTTP_200_OK)
async def delete_file(file_id: str, db: AsyncSession = Depends(get_db)) -> DeleteFileResponse:
    """
    Delete a file.

    Args:
        file_id (str): The ID of the file to delete.
        db (AsyncSession): The SQLAlchemy asynchronous session, injected via dependency.

    Returns:
        DeleteFileResponse: A confirmation response.

    Raises:
        HTTPException: If the file is not found.
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
    Returns the contents of the specified file.

    Args:
        file_id (str): The ID of the file to retrieve content for.
        db (AsyncSession): The SQLAlchemy asynchronous session, injected via dependency.

    Returns:
        Response: The content of the file.

    Raises:
        HTTPException: If the file is not found or content is missing.
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
