"""
FastAPI router for OpenAI-compatible Vector Stores API endpoints.

This module provides an implementation of OpenAI's Vector Stores API,
using OVOS text embedding plugins and a configurable EmbeddingsDB backend.
It uses SQLAlchemy ORM for metadata storage.
"""
import inspect
import json
import os
import random
import string
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, AsyncGenerator, List

from fastapi import APIRouter, HTTPException, status, Query, Depends, FastAPI
from ovos_plugin_manager.embeddings import load_embeddings_db_plugin
from ovos_plugin_manager.templates.embeddings import EmbeddingsDB
from ovos_utils.log import LOG
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from ovos_persona_server.config import settings
from ovos_persona_server.embeddings import get_text_embeddings, TextEmbedder
from ovos_persona_server.metadata import get_async_db, VectorStore as VectorStoreORM, \
    VectorStoreFile as VectorStoreFileORM, File as FileORM, FileChunk as FileChunkORM
from ovos_persona_server.schemas.openai_vectorstore import (
    VectorStoreObject, CreateVectorStoreRequest, UpdateVectorStoreRequest, ListVectorStoresResponse,
    VectorStoreFileObject, CreateVectorStoreFileRequest, ListVectorStoreFilesResponse, SearchVectorStoreFileRequest,
    VectorStoreFileCounts, VectorStoreDeleted, VectorStoreFileDeleted, VectorStoreSearchResponse, SearchResultChunk,
    VectorStoreFileLastError
)

# A single vectorDB interface is shared
# vector_store_id corresponds to a collection inside the EmbeddingsDB
_vector_db_instance: Optional[EmbeddingsDB] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages the lifespan of the FastAPI application, initializing the global vector database
    and text embedder instances.
    """
    embedder = await get_text_embeddings()
    await get_vector_db(embedder)
    yield
    # No specific shutdown logic needed for these dependencies currently


vector_stores_router = APIRouter(prefix="/openai/v1/vector_stores", tags=["vector_stores"])


def _generate_id(prefix: str) -> str:
    """Generates a unique ID with a given prefix."""
    return f"{prefix}_{''.join(random.choices(string.ascii_letters + string.digits, k=24))}"


def _get_current_timestamp() -> int:
    """Returns the current Unix timestamp."""
    return int(time.time())


async def get_vector_db(embedder: TextEmbedder = Depends(get_text_embeddings)) -> EmbeddingsDB:
    """
    FastAPI dependency that provides the initialized EmbeddingsDB instance.
    """
    global _vector_db_instance
    if _vector_db_instance is None:
        cfg = settings.embeddings_db_config
        try:
            test_embedding = embedder.get_embeddings("test")
            cfg["vector_size"] = len(test_embedding)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Text embedder failed to provide a test embedding for vector size determination: {e}") from e

        db_plugin_class = load_embeddings_db_plugin(settings.embeddings_db_plugin)
        # EmbeddingsDB plugins have heterogeneous constructors: the base template
        # and ovos-qdrant take ``config=``, while ovos-chromadb takes a positional
        # ``path``. Adapt to whichever the plugin actually accepts.
        init_params = inspect.signature(db_plugin_class.__init__).parameters
        if "config" in init_params:
            _vector_db_instance = db_plugin_class(config=cfg)
        elif "path" in init_params:
            _vector_db_instance = db_plugin_class(path=cfg["path"])
        else:
            _vector_db_instance = db_plugin_class()
        LOG.debug(f"Initialized EmbeddingsDB plugin: {settings.embeddings_db_plugin} with config: {cfg}")
    return _vector_db_instance


# --- Helper Functions ---

async def _get_vector_store_orm_or_404(vector_store_id: str, db: AsyncSession) -> VectorStoreORM:
    """Retrieve a vector store ORM object from DB or raise HTTPException 404."""
    result = await db.execute(select(VectorStoreORM).where(VectorStoreORM.id == vector_store_id))
    store_orm = result.scalars().first()
    if not store_orm:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vector store not found.")
    return store_orm


async def _get_vector_store_object_with_counts(vector_store_orm: VectorStoreORM, db: AsyncSession) -> VectorStoreObject:
    """
    Helper to construct VectorStoreObject with dynamic file counts and usage bytes.
    """
    # Fetch file counts
    file_counts_query = select(
        VectorStoreFileORM.status,
        func.count(VectorStoreFileORM.id).label("count")
    ).where(VectorStoreFileORM.vector_store_id == vector_store_orm.id).group_by(VectorStoreFileORM.status)

    file_counts_result = await db.execute(file_counts_query)
    counts_rows = file_counts_result.all()
    counts_dict = {row.status: row.count for row in counts_rows}

    file_counts = VectorStoreFileCounts(
        in_progress=counts_dict.get("in_progress", 0),
        completed=counts_dict.get("completed", 0),
        failed=counts_dict.get("failed", 0),
        cancelled=counts_dict.get("cancelled", 0),
        total=sum(counts_dict.values())
    )

    # Calculate usage_bytes
    usage_bytes_query = select(func.sum(VectorStoreFileORM.usage_bytes)).where(VectorStoreFileORM.vector_store_id == vector_store_orm.id)
    usage_bytes_result = await db.execute(usage_bytes_query)
    usage_bytes = usage_bytes_result.scalar_one_or_none() or 0

    return VectorStoreObject(**vector_store_orm.to_dict(file_counts=file_counts, usage_bytes=usage_bytes))


def _chunk_text(text: str, max_chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Simple text chunking based on character count.
    NOTE: This is a basic substitute for token-based chunking.
    """
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk_size
        chunks.append(text[start:end])
        start += max_chunk_size - chunk_overlap
        if start >= len(text):
            break
    return chunks


# --- Vector Store Endpoints ---

@vector_stores_router.post("/", response_model=VectorStoreObject, status_code=status.HTTP_201_CREATED)
async def create_vector_store(
        request: CreateVectorStoreRequest,
        db: AsyncSession = Depends(get_async_db),
        db_embeddings: EmbeddingsDB = Depends(get_vector_db)
) -> VectorStoreObject:
    """
    Create a new vector store.
    """
    vs_id = _generate_id("vs")
    created_at = _get_current_timestamp()

    # Handle expiration_after
    expires_after_anchor = None
    expires_after_minutes = None
    expires_at = None
    if request.expires_after:
        expires_after_anchor = request.expires_after.anchor
        expires_after_minutes = request.expires_after.minutes
        # Calculate expires_at based on last_active_at (which is created_at initially)
        expires_at = created_at + expires_after_minutes * 60

    # Create collection in the vector database
    try:
        db_embeddings.create_collection(name=vs_id, metadata=request.metadata)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to create collection in vector DB: {e}") from e

    # Create ORM object for VectorStore
    new_vector_store_orm = VectorStoreORM(
        id=vs_id,
        object="vector_store",
        created_at=created_at,
        name=request.name,
        status="completed",  # Initial status
        expires_after_anchor=expires_after_anchor,
        expires_after_minutes=expires_after_minutes,
        expires_at=expires_at,
        last_active_at=created_at,
        extra_metadata=json.dumps(request.metadata) if request.metadata else None # Use extra_metadata here
    )

    # Insert metadata into SQLite via SQLAlchemy
    try:
        db.add(new_vector_store_orm)
        await db.commit()
        await db.refresh(new_vector_store_orm)  # Refresh to get any default values
    except Exception as e:
        # Rollback vector DB operation if metadata insertion fails
        db_embeddings.delete_collection(name=vs_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {e}") from e

    # TODO: Handle initial file_ids if provided (ideally as a background task)

    # Return the created VectorStoreObject with initial counts
    return await _get_vector_store_object_with_counts(new_vector_store_orm, db)


@vector_stores_router.get("/", response_model=ListVectorStoresResponse)
async def list_vector_stores(
        db: AsyncSession = Depends(get_async_db),
        limit: int = Query(20, ge=1, le=100),
        order: str = Query("desc", pattern="^(asc|desc)$"),
        after: Optional[str] = None,
        before: Optional[str] = None
) -> ListVectorStoresResponse:
    """
    List all vector stores.
    """
    query = select(VectorStoreORM)

    # Apply ordering
    if order == "desc":
        query = query.order_by(VectorStoreORM.created_at.desc())
    else:
        query = query.order_by(VectorStoreORM.created_at.asc())

    # Fetch all relevant vector stores first to handle pagination logic in Python
    # For large datasets, this should be optimized with database-level pagination
    result = await db.execute(query)
    all_vector_stores_orm = result.scalars().all()

    all_vector_stores: List[VectorStoreObject] = []
    for vs_orm in all_vector_stores_orm:
        all_vector_stores.append(await _get_vector_store_object_with_counts(vs_orm, db))

    # Apply pagination logic
    start_index: int = 0
    end_index: int = len(all_vector_stores)

    if after:
        try:
            after_index: int = next(i for i, vs in enumerate(all_vector_stores) if vs.id == after)
            start_index = after_index + 1
        except StopIteration:
            pass

    if before:
        try:
            before_index: int = next(i for i, vs in enumerate(all_vector_stores) if vs.id == before)
            end_index = before_index
        except StopIteration:
            pass

    paginated_stores: List[VectorStoreObject] = all_vector_stores[start_index:end_index]
    data: List[VectorStoreObject] = paginated_stores[:limit]
    has_more: bool = len(paginated_stores) > limit

    first_id: Optional[str] = data[0].id if data else None
    last_id: Optional[str] = data[-1].id if data else None

    return ListVectorStoresResponse(
        object="list",
        data=data,
        first_id=first_id,
        last_id=last_id,
        has_more=has_more
    )


@vector_stores_router.get("/{vector_store_id}", response_model=VectorStoreObject)
async def retrieve_vector_store(
        vector_store_id: str,
        db: AsyncSession = Depends(get_async_db)
) -> VectorStoreObject:
    """
    Retrieve a specific vector store by its ID.
    """
    vector_store_orm = await _get_vector_store_orm_or_404(vector_store_id, db)
    return await _get_vector_store_object_with_counts(vector_store_orm, db)


@vector_stores_router.post("/{vector_store_id}", response_model=VectorStoreObject)
async def modify_vector_store(
        vector_store_id: str,
        request: UpdateVectorStoreRequest,
        db: AsyncSession = Depends(get_async_db)
) -> VectorStoreObject:
    """
    Modify an existing vector store.
    """
    vector_store_orm = await _get_vector_store_orm_or_404(vector_store_id, db)

    update_data = request.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update.")

    # Handle special fields for database storage
    if 'metadata' in update_data:
        vector_store_orm.extra_metadata = json.dumps(update_data['metadata'])
    if 'name' in update_data:
        vector_store_orm.name = update_data['name']

    if 'expires_after' in update_data:
        expires_after_data: Dict[str, Any] = update_data['expires_after']
        if expires_after_data:
            vector_store_orm.expires_after_anchor = expires_after_data["anchor"]
            vector_store_orm.expires_after_minutes = expires_after_data["minutes"]
            # Recalculate expires_at based on current last_active_at
            # No need to re-fetch, use the existing ORM object's last_active_at
            last_active_at = vector_store_orm.last_active_at or _get_current_timestamp()
            vector_store_orm.expires_at = last_active_at + expires_after_data["minutes"] * 60
        else:
            vector_store_orm.expires_after_anchor = None
            vector_store_orm.expires_after_minutes = None
            vector_store_orm.expires_at = None

    await db.commit()
    await db.refresh(vector_store_orm)  # Refresh to get updated values

    return await _get_vector_store_object_with_counts(vector_store_orm, db)


@vector_stores_router.delete("/{vector_store_id}", response_model=VectorStoreDeleted)
async def delete_vector_store(
        vector_store_id: str,
        db: AsyncSession = Depends(get_async_db),
        db_embeddings: EmbeddingsDB = Depends(get_vector_db)
) -> VectorStoreDeleted:
    """
    Delete a vector store.
    """
    vector_store_orm = await _get_vector_store_orm_or_404(vector_store_id, db)

    # Delete from vector DB
    try:
        db_embeddings.delete_collection(name=vector_store_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Warning: Failed to delete collection '{vector_store_id}' from vector DB: {e}") from e

    # Delete from metadata DB (cascades to vector_store_files due to ForeignKey ondelete="CASCADE")
    await db.delete(vector_store_orm)
    await db.commit()

    return VectorStoreDeleted(id=vector_store_id, object="vector_store.deleted", deleted=True)


# --- Vector Store File Endpoints ---

@vector_stores_router.post("/{vector_store_id}/files", response_model=VectorStoreFileObject, status_code=status.HTTP_201_CREATED)
async def create_vector_store_file(
        vector_store_id: str,
        request: CreateVectorStoreFileRequest,
        db: AsyncSession = Depends(get_async_db),
        db_embeddings: EmbeddingsDB = Depends(get_vector_db),
        embedder: TextEmbedder = Depends(get_text_embeddings)
) -> VectorStoreFileObject:
    """
    Create a vector store file. This chunks and embeds the file content,
    adding it to the vector store.
    Prevents adding the same file to the same vector store more than once.
    """
    await _get_vector_store_orm_or_404(vector_store_id, db)

    # 1. Check if this file is already associated with this vector store
    existing_vs_file_query = select(VectorStoreFileORM).where(
        VectorStoreFileORM.vector_store_id == vector_store_id,
        VectorStoreFileORM.file_id == request.file_id
    )
    existing_vs_file_result = await db.execute(existing_vs_file_query)
    existing_vs_file_orm = existing_vs_file_result.scalars().first()

    if existing_vs_file_orm:
        # If an existing association is found, return it (idempotency)
        LOG.debug(f"File '{request.file_id}' is already associated with vector store '{vector_store_id}'. "
                  f"Returning existing entry.")
        return VectorStoreFileObject(**existing_vs_file_orm.to_dict())

    # 2. Get file metadata and content from the 'files' table using ORM
    file_result = await db.execute(select(FileORM).where(FileORM.id == request.file_id))
    file_orm = file_result.scalars().first()
    if not file_orm:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File with id '{request.file_id}' not found.")

    vsf_id = _generate_id("vsf")
    created_at = _get_current_timestamp()

    # Determine chunking strategy for storage
    chunking_strategy_json = None
    if request.chunking_strategy:
        chunking_strategy_json = request.chunking_strategy.model_dump_json()

    # 3. Create ORM object for VectorStoreFile and insert initial record with status 'in_progress'
    new_vs_file_orm = VectorStoreFileORM(
        id=vsf_id,
        object="vector_store.file",
        vector_store_id=vector_store_id,
        file_id=request.file_id,
        created_at=created_at,
        status="in_progress",
        usage_bytes=file_orm.bytes,
        chunking_strategy=chunking_strategy_json
    )
    db.add(new_vs_file_orm)
    await db.commit()
    await db.refresh(new_vs_file_orm)

    try:
        # 4. Read file content from the ORM object's content_data or local_path
        content_bytes: Optional[bytes] = None
        if file_orm.local_path and os.path.exists(file_orm.local_path):
            with open(file_orm.local_path, "rb") as f:
                content_bytes = f.read()
        elif file_orm.content_data:
            content_bytes = file_orm.content_data

        if content_bytes is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="File content not found for embedding.")

        # Assuming text content for embedding. If binary, needs decoding or specific handling.
        # For simplicity, assuming UTF-8 decodable text for chunking/embedding.
        try:
            text_content = content_bytes.decode('utf-8')
        except UnicodeDecodeError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="File content is not valid UTF-8 text. Only text files are supported for embedding.") from e
        # 5. Chunk the content
        chunks: List[str]
        # TODO - support for chunking strategies - currently uses default
        chunks = _chunk_text(text_content, 800, 400)  # Default values for auto-chunking

        # 6. Get embeddings for each chunk and store chunk metadata
        if chunks:
            embeddings_list = [embedder.get_embeddings(c) for c in chunks]
            chunk_orms = []
            embedding_keys_for_batch = []
            embedding_metadata_for_batch = []

            for i, chunk_content in enumerate(chunks):
                chunk_id = f"{new_vs_file_orm.id}_chunk_{i}" # Unique ID for this chunk
                embedding_key = f"{vector_store_id}_{file_orm.id}_chunk_{i}" # Key for EmbeddingsDB

                chunk_orms.append(FileChunkORM(
                    id=chunk_id,
                    file_id=file_orm.id,
                    vector_store_file_id=new_vs_file_orm.id,
                    chunk_index=i,
                    chunk_content=chunk_content,
                    embedding_key=embedding_key
                ))
                embedding_keys_for_batch.append(embedding_key)
                embedding_metadata_for_batch.append({
                    "file_id": file_orm.id,
                    "chunk_index": i,
                    "content": chunk_content,
                    "vector_store_file_id": new_vs_file_orm.id
                })

            # Add all chunk ORMs to the session
            db.add_all(chunk_orms)
            await db.commit() # Commit chunks to DB before adding to vector DB for consistency
            await db.refresh(new_vs_file_orm) # Refresh to ensure relationship is loaded if needed

            # Add embeddings to the vector DB collection
            # Assuming EmbeddingsDB.add_embeddings_batch takes keys, embeddings, metadata, collection_name
            db_embeddings.add_embeddings_batch(
                keys=embedding_keys_for_batch,
                embeddings=embeddings_list,
                metadata=embedding_metadata_for_batch,
                collection_name=vector_store_id
            )

        # 8. Update status to 'completed'
        new_vs_file_orm.status = "completed"
        await db.commit()
        await db.refresh(new_vs_file_orm)

    except Exception as e:
        # On failure, update status and store error
        error_info = VectorStoreFileLastError(code="server_error", message=str(e)).model_dump_json()
        new_vs_file_orm.status = "failed"
        new_vs_file_orm.last_error = error_info
        await db.commit()
        await db.refresh(new_vs_file_orm)
        # Re-raise to inform the client of the failure
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process file: {e}") from e

    # Return the newly created VectorStoreFileObject
    return VectorStoreFileObject(**new_vs_file_orm.to_dict())


@vector_stores_router.get("/{vector_store_id}/files", response_model=ListVectorStoreFilesResponse)
async def list_vector_store_files(
        vector_store_id: str,
        db: AsyncSession = Depends(get_async_db),
        limit: int = Query(20, ge=1, le=100),
        order: str = Query("desc", pattern="^(asc|desc)$"),
        after: Optional[str] = None,
        before: Optional[str] = None
) -> ListVectorStoreFilesResponse:
    """List files associated with a vector store."""
    await _get_vector_store_orm_or_404(vector_store_id, db)

    query = select(VectorStoreFileORM).where(VectorStoreFileORM.vector_store_id == vector_store_id)

    # Apply ordering
    if order == "desc":
        query = query.order_by(VectorStoreFileORM.created_at.desc())
    else:
        query = query.order_by(VectorStoreFileORM.created_at.asc())

    result = await db.execute(query)
    all_vs_files_orm = result.scalars().all()

    all_vector_store_files: List[VectorStoreFileObject] = []
    for vs_file_orm in all_vs_files_orm:
        all_vector_store_files.append(VectorStoreFileObject(**vs_file_orm.to_dict()))

    # Apply pagination logic
    start_index: int = 0
    end_index: int = len(all_vector_store_files)

    if after:
        try:
            after_index: int = next(i for i, vsf in enumerate(all_vector_store_files) if vsf.id == after)
            start_index = after_index + 1
        except StopIteration:
            pass

    if before:
        try:
            before_index: int = next(i for i, vsf in enumerate(all_vector_store_files) if vsf.id == before)
            end_index = before_index
        except StopIteration:
            pass

    paginated_files: List[VectorStoreFileObject] = all_vector_store_files[start_index:end_index]
    data: List[VectorStoreFileObject] = paginated_files[:limit]
    has_more: bool = len(paginated_files) > limit

    first_id: Optional[str] = data[0].id if data else None
    last_id: Optional[str] = data[-1].id if data else None

    return ListVectorStoreFilesResponse(
        object="list",
        data=data,
        first_id=first_id,
        last_id=last_id,
        has_more=has_more
    )


@vector_stores_router.get("/{vector_store_id}/files/{file_id}", response_model=VectorStoreFileObject)
async def retrieve_vector_store_file(
        vector_store_id: str,
        file_id: str,
        db: AsyncSession = Depends(get_async_db)
) -> VectorStoreFileObject:
    """Retrieve a specific file from a vector store."""
    await _get_vector_store_orm_or_404(vector_store_id, db)

    result = await db.execute(
        select(VectorStoreFileORM).where(
            VectorStoreFileORM.vector_store_id == vector_store_id,
            VectorStoreFileORM.file_id == file_id
        )
    )
    vs_file_orm = result.scalars().first()
    if not vs_file_orm:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vector store file not found.")

    return VectorStoreFileObject(**vs_file_orm.to_dict())


@vector_stores_router.delete("/{vector_store_id}/files/{file_id}", response_model=VectorStoreFileDeleted)
async def delete_vector_store_file(
        vector_store_id: str,
        file_id: str,
        db: AsyncSession = Depends(get_async_db),
        db_embeddings: EmbeddingsDB = Depends(get_vector_db)
) -> VectorStoreFileDeleted:
    """
    Delete a file from a vector store.
    This also deletes associated chunk embeddings from the vector DB
    and chunk metadata from the local DB.
    """
    result = await db.execute(
        select(VectorStoreFileORM).where(
            VectorStoreFileORM.vector_store_id == vector_store_id,
            VectorStoreFileORM.file_id == file_id
        )
    )
    vs_file_orm = result.scalars().first()
    if not vs_file_orm:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vector store file not found.")

    try:
        # 1. Find all chunk_ids (embedding_keys) associated with this VectorStoreFile
        chunk_keys_query = select(FileChunkORM.embedding_key).where(
            FileChunkORM.vector_store_file_id == vs_file_orm.id
        )
        chunk_keys_result = await db.execute(chunk_keys_query)
        embedding_keys_to_delete = [row.embedding_key for row in chunk_keys_result.scalars().all()]

        # 2. Delete embeddings from the external vector database
        if embedding_keys_to_delete:
            try:
                db_embeddings.delete_embeddings_batch(keys=embedding_keys_to_delete,
                                                      collection_name=vector_store_id)

            except Exception as e:
                # Both databases must remain in sync
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    detail=f"Failed to delete embeddings from vector DB: {e}") from e
        # 3. Delete chunk metadata from the local database (cascades from VectorStoreFile deletion if relationship is set up correctly)
        # However, for explicit control and clarity, we'll delete them directly here before deleting the parent.
        await db.execute(delete(FileChunkORM).where(FileChunkORM.vector_store_file_id == vs_file_orm.id))

        # 4. Delete the VectorStoreFile record from the metadata database
        await db.delete(vs_file_orm)
        await db.commit()

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to delete vector store file and its chunks: {e}") from e

    return VectorStoreFileDeleted(id=file_id, object="vector_store.file.deleted", deleted=True)


# --- Vector Store Search Endpoint ---

@vector_stores_router.post("/{vector_store_id}/search", response_model=VectorStoreSearchResponse)
async def search_vector_store(
        vector_store_id: str,
        request: SearchVectorStoreFileRequest,
        db: AsyncSession = Depends(get_async_db),
        db_embeddings: EmbeddingsDB = Depends(get_vector_db),
        embedder: TextEmbedder = Depends(get_text_embeddings)
) -> VectorStoreSearchResponse:
    """
    Search a vector store for relevant chunks.
    """
    await _get_vector_store_orm_or_404(vector_store_id, db)

    try:
        # 1. Embed the search query
        query_embedding = embedder.get_embeddings(request.query)

        # 2. Query the vector database
        # TODO: Implement filtering based on request.filters
        search_results = db_embeddings.query(
            embeddings=query_embedding,
            top_k=request.max_num_results,
            collection_name=vector_store_id,
            return_metadata=True
        )

        # 3. Format the results
        data = []
        for _key, score, metadata in search_results:
            # Note: content needs to be retrieved from FileChunkORM
            if metadata and 'file_id' in metadata:
                data.append(SearchResultChunk(
                    type="file_search",  # Explicitly set type as per schema
                    file_id=metadata['file_id'],
                    content=metadata['content'],
                    metadata=metadata,
                    score=score
                ))

        return VectorStoreSearchResponse(object="list", data=data,
                                         has_more=False)  # has_more is always False in this basic impl

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e}"
        ) from e
