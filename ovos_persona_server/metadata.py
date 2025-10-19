"""
SQLAlchemy database setup and ORM models for the OVOS Persona Server.

This module defines the asynchronous SQLAlchemy engine, session management,
and declarative base for defining ORM models. It also includes the ORM
definitions for 'files', 'vector_stores', and 'vector_store_files' tables,
mirroring the OpenAI API's data structures.
"""

import json
import os
from typing import AsyncGenerator, Dict, Optional

from ovos_utils.log import LOG
from sqlalchemy import Column, Integer, String, ForeignKey, Text, LargeBinary
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

from ovos_persona_server.config import settings
from ovos_persona_server.schemas.openai_files import FilePurpose
from ovos_persona_server.schemas.openai_vectorstore import (
    VectorStoreFileStatus, VectorStoreStatus, VectorStoreExpirationAfter,
    VectorStoreFileLastError, VectorStoreFileChunkingStrategy
)

# Define the single, unified database file path
UNIFIED_DATABASE_FILE = os.path.join(settings.file_storage_path, "metadata.db")

# Ensure the directory for the database file exists
os.makedirs(os.path.dirname(UNIFIED_DATABASE_FILE), exist_ok=True)

# SQLAlchemy Async Engine
# Using 'sqlite+aiosqlite' for asynchronous SQLite operations
engine = create_async_engine(f"sqlite+aiosqlite:///{UNIFIED_DATABASE_FILE}", echo=False)

# Asynchronous Session Local
AsyncSessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False  # Important for keeping objects accessible after commit
)

Base = declarative_base()


# --- ORM Models ---

class File(Base):
    """
    SQLAlchemy ORM model for the 'files' table, representing uploaded files.
    File content can be stored on disk, in the database, or both, based on settings.
    """
    __tablename__ = "files"

    id = Column(String, primary_key=True, index=True)
    object = Column(String, nullable=False, default="file")
    bytes = Column(Integer, nullable=False)
    created_at = Column(Integer, nullable=False, default=func.unixepoch())
    filename = Column(String, nullable=False)
    purpose = Column(String, nullable=False)  # Stored as string, but maps to FilePurpose enum
    status = Column(String, nullable=False, default="uploaded")
    status_details = Column(String, nullable=True)
    local_path = Column(String, nullable=True) # Path to file on disk if stored on disk
    content_data = Column(LargeBinary, nullable=True) # Binary content if stored in database
    content_hash = Column(String, nullable=True, unique=False) # Store content hash to prevent duplicates

    # Relationship to VectorStoreFile (one-to-many)
    vector_store_files = relationship("VectorStoreFile", back_populates="file_obj")
    # Relationship to FileChunk (one-to-many) - a file can have many chunks
    file_chunks = relationship("FileChunk", back_populates="file_obj")


    def to_dict(self):
        """
        Serialize the File ORM record to a dictionary matching the external file schema.
        
        Returns:
            dict: Dictionary with keys:
                - id (str)
                - object (str)
                - bytes (int)
                - created_at (int)
                - filename (str)
                - purpose (FilePurpose): the purpose as a FilePurpose enum
                - status (str)
                - status_details (str | None)
        """
        return {
            "id": self.id,
            "object": self.object,
            "bytes": self.bytes,
            "created_at": self.created_at,
            "filename": self.filename,
            "purpose": FilePurpose(self.purpose),  # Convert back to enum
            "status": self.status,
            "status_details": self.status_details
        }


class VectorStore(Base):
    """
    SQLAlchemy ORM model for the 'vector_stores' table.
    """
    __tablename__ = "vector_stores"

    id = Column(String, primary_key=True, index=True)
    object = Column(String, nullable=False, default="vector_store")
    created_at = Column(Integer, nullable=False, default=func.unixepoch())
    name = Column(String, nullable=True)
    status = Column(String, nullable=False, default="completed")  # Maps to VectorStoreStatus enum

    # Expiration fields
    expires_after_anchor = Column(String, nullable=True)
    expires_after_minutes = Column(Integer, nullable=True)
    expires_at = Column(Integer, nullable=True)
    last_active_at = Column(Integer, nullable=True)

    # Metadata stored as JSON string
    extra_metadata = Column(Text, nullable=True) # Use Text for potentially long JSON strings

    # Relationship to VectorStoreFile (one-to-many)
    vector_store_files = relationship("VectorStoreFile", back_populates="vector_store_obj")

    def to_dict(self, file_counts: Optional[Dict[str, int]] = None, usage_bytes: int = 0):
        """
        Serialize the VectorStore ORM instance into a dictionary matching the public API schema.
        
        Parameters:
            file_counts (Optional[Dict[str, int]]): Optional mapping of file categories to their counts to include in the output.
            usage_bytes (int): Total usage in bytes to include in the output.
        
        Returns:
            dict: Dictionary with keys:
                - id: the vector store identifier.
                - object: the object type string.
                - created_at: creation timestamp (epoch seconds).
                - name: the vector store name or None.
                - usage_bytes: the provided usage_bytes value.
                - file_counts: the provided file_counts mapping or None.
                - status: `VectorStoreStatus` enum constructed from the stored status.
                - expires_after: a `VectorStoreExpirationAfter` object when anchor and minutes are set, otherwise None.
                - expires_at: explicit expiration timestamp or None.
                - last_active_at: last active timestamp or None.
                - metadata: parsed JSON from `extra_metadata` or None.
        """
        expires_after_obj = None
        if self.expires_after_anchor and self.expires_after_minutes is not None:
            expires_after_obj = VectorStoreExpirationAfter(
                anchor=self.expires_after_anchor,
                minutes=self.expires_after_minutes
            )

        return {
            "id": self.id,
            "object": self.object,
            "created_at": self.created_at,
            "name": self.name,
            "usage_bytes": usage_bytes,
            "file_counts": file_counts,  # This needs to be passed dynamically
            "status": VectorStoreStatus(self.status),  # Convert back to enum
            "expires_after": expires_after_obj,
            "expires_at": self.expires_at,
            "last_active_at": self.last_active_at,
            "metadata": json.loads(self.extra_metadata) if self.extra_metadata else None,
        }


class VectorStoreFile(Base):
    """
    SQLAlchemy ORM model for the 'vector_store_files' table,
    linking files to vector stores.
    """
    __tablename__ = "vector_store_files"

    id = Column(String, primary_key=True, index=True)
    object = Column(String, nullable=False, default="vector_store.file")
    vector_store_id = Column(String, ForeignKey("vector_stores.id", ondelete="CASCADE"), nullable=False)
    file_id = Column(String, ForeignKey("files.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(Integer, nullable=False, default=func.unixepoch())
    status = Column(String, nullable=False, default="in_progress")  # Maps to VectorStoreFileStatus enum
    last_error = Column(Text, nullable=True)  # Stored as JSON string of VectorStoreFileLastError
    chunking_strategy = Column(Text, nullable=True)  # Stored as JSON string of VectorStoreFileChunkingStrategy
    usage_bytes = Column(Integer, nullable=False)

    # Relationships
    vector_store_obj = relationship("VectorStore", back_populates="vector_store_files")
    file_obj = relationship("File", back_populates="vector_store_files")
    # Relationship to FileChunk (one-to-many) - a vector store file can have many chunks
    chunks = relationship("FileChunk", back_populates="vector_store_file_obj", cascade="all, delete-orphan")


    def to_dict(self):
        """
        Serialize the VectorStoreFile ORM instance into a dictionary matching the OpenAI-style API shape.
        
        Returns:
            dict: Mapping with keys:
                - id (str): Original `file_id` from the association.
                - object (str): The object's type string.
                - created_at (int): Unix epoch creation timestamp.
                - vector_store_id (str): Associated vector store id.
                - status (VectorStoreFileStatus): Status converted to the enum.
                - usage_bytes (int): Stored usage in bytes.
                - last_error (VectorStoreFileLastError | None): Parsed `last_error` JSON as a Pydantic object, or `None` if absent or malformed.
                - chunking_strategy (VectorStoreFileChunkingStrategy | None): Parsed `chunking_strategy` JSON reconstructed into the Pydantic union type, or `None` if absent or malformed.
        """
        last_error_obj = None
        if self.last_error:
            try:
                error_data = json.loads(self.last_error)
                last_error_obj = VectorStoreFileLastError(**error_data)
            except (json.JSONDecodeError, TypeError):
                pass  # Handle malformed JSON gracefully

        chunking_strategy_obj = None
        if self.chunking_strategy:
            try:
                strategy_data = json.loads(self.chunking_strategy)
                # Reconstruct the Pydantic Union type
                chunking_strategy_obj = VectorStoreFileChunkingStrategy(strategy_data)
            except (json.JSONDecodeError, TypeError):
                pass  # Handle malformed JSON gracefully

        return {
            "id": self.file_id,  # OpenAI API returns the original file_id here
            "object": self.object,
            "created_at": self.created_at,
            "vector_store_id": self.vector_store_id,
            "status": VectorStoreFileStatus(self.status),  # Convert back to enum
            "usage_bytes": self.usage_bytes,
            "last_error": last_error_obj,
            "chunking_strategy": chunking_strategy_obj,
        }


class FileChunk(Base):
    """
    SQLAlchemy ORM model for the 'file_chunks' table,
    storing metadata about individual chunks of files.
    """
    __tablename__ = "file_chunks"

    id = Column(String, primary_key=True, index=True) # Unique ID for the chunk
    file_id = Column(String, ForeignKey("files.id", ondelete="CASCADE"), nullable=False)
    vector_store_file_id = Column(String, ForeignKey("vector_store_files.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_content = Column(Text, nullable=False) # Store the actual text content of the chunk
    embedding_key = Column(String, nullable=False, unique=True) # Key used in the EmbeddingsDB

    # Relationships
    file_obj = relationship("File", back_populates="file_chunks")
    vector_store_file_obj = relationship("VectorStoreFile", back_populates="chunks")


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides an asynchronous SQLAlchemy database session.
    This session is automatically closed after the request is processed.

    Yields:
        AsyncSession: An active SQLAlchemy asynchronous session.
    """
    async with AsyncSessionLocal() as session:
        yield session
        await session.close()


async def init_db():
    """
    Create all ORM tables defined on Base.metadata in the configured database.
    
    This connects to the engine and runs metadata.create_all to ensure every table declared by the declarative models exists.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    LOG.debug("Database tables created/verified.")