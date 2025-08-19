import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column, String, Text, DateTime, Integer, Float, 
    Boolean, ForeignKey, Index, JSON, LargeBinary
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Basic information
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Content information
    title: Mapped[Optional[str]] = mapped_column(String(500))
    description: Mapped[Optional[str]] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(10), default="en")
    
    # Processing information
    status: Mapped[str] = mapped_column(
        String(20), 
        default="pending",
        nullable=False
    )  # pending, processing, completed, failed
    processing_started: Mapped[Optional[datetime]] = mapped_column(DateTime)
    processing_completed: Mapped[Optional[datetime]] = mapped_column(DateTime)
    processing_error: Mapped[Optional[str]] = mapped_column(Text)
    
    # Metadata
    metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    tags: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_documents_status", "status"),
        Index("idx_documents_file_type", "file_type"),
        Index("idx_documents_created_at", "created_at"),
        Index("idx_documents_filename", "filename"),
    )


class DocumentChunk(Base):
    """Chunk model for storing document text segments with embeddings."""
    
    __tablename__ = "document_chunks"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Foreign keys
    document_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Chunk information
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_length: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Embedding information
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_vector_id: Mapped[Optional[str]] = mapped_column(String(255))
    embedding_dimensions: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Semantic information
    summary: Mapped[Optional[str]] = mapped_column(Text)
    keywords: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Metadata
    metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_chunks_document_id", "document_id"),
        Index("idx_chunks_embedding_model", "embedding_model"),
        Index("idx_chunks_content_length", "content_length"),
        Index("idx_chunks_created_at", "created_at"),
    )


class VectorCollection(Base):
    """Model for managing vector collections in the vector database."""
    
    __tablename__ = "vector_collections"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Collection information
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Vector configuration
    vector_size: Mapped[int] = mapped_column(Integer, nullable=False)
    distance_metric: Mapped[str] = mapped_column(
        String(50), 
        default="cosine",
        nullable=False
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    vector_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Metadata
    metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_collections_name", "name"),
        Index("idx_collections_active", "is_active"),
    )


class QueryLog(Base):
    """Model for logging user queries and responses."""
    
    __tablename__ = "query_logs"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Query information
    query: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String(50), default="rag")
    
    # Response information
    response: Mapped[Optional[str]] = mapped_column(Text)
    response_length: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Performance metrics
    processing_time: Mapped[Optional[float]] = mapped_column(Float)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Source information
    source_documents: Mapped[List[str]] = mapped_column(JSON, default=list)
    similarity_scores: Mapped[List[float]] = mapped_column(JSON, default=list)
    
    # User information
    user_id: Mapped[Optional[str]] = mapped_column(String(255))
    session_id: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Metadata
    metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=func.now(),
        nullable=False
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_query_logs_created_at", "created_at"),
        Index("idx_query_logs_user_id", "user_id"),
        Index("idx_query_logs_query_type", "query_type"),
    )


class EmbeddingModel(Base):
    """Model for tracking embedding models and their usage."""
    
    __tablename__ = "embedding_models"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Model information
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Model specifications
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    max_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    supported_languages: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Performance metrics
    average_processing_time: Mapped[Optional[float]] = mapped_column(Float)
    total_embeddings_generated: Mapped[int] = mapped_column(
        Integer, 
        default=0,
        nullable=False
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Metadata
    metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_models_name", "name"),
        Index("idx_models_provider", "provider"),
        Index("idx_models_active", "is_active"),
    )


# Pydantic models for API responses
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class DocumentBase(BaseModel):
    """Base document model for API requests/responses."""
    filename: str = Field(..., description="Name of the file")
    file_type: str = Field(..., description="Type of the file")
    file_size: int = Field(..., description="Size of the file in bytes")
    title: Optional[str] = Field(None, description="Document title")
    description: Optional[str] = Field(None, description="Document description")
    language: str = Field(default="en", description="Document language")
    tags: List[str] = Field(default_factory=list, description="Document tags")


class DocumentCreate(DocumentBase):
    """Model for creating a new document."""
    pass


class DocumentResponse(DocumentBase):
    """Model for document API responses."""
    id: str
    status: str
    created_at: datetime
    updated_at: datetime
    chunk_count: Optional[int] = None
    
    class Config:
        from_attributes = True


class ChunkBase(BaseModel):
    """Base chunk model for API requests/responses."""
    content: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., description="Index of the chunk in the document")
    summary: Optional[str] = Field(None, description="Chunk summary")
    keywords: List[str] = Field(default_factory=list, description="Chunk keywords")


class ChunkResponse(ChunkBase):
    """Model for chunk API responses."""
    id: str
    document_id: str
    content_length: int
    embedding_model: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    """Model for query API requests."""
    query: str = Field(..., description="User query")
    query_type: str = Field(default="rag", description="Type of query")
    top_k: int = Field(default=5, description="Number of top results to return")
    similarity_threshold: float = Field(
        default=0.7, 
        description="Minimum similarity threshold"
    )
    include_metadata: bool = Field(default=True, description="Include metadata in response")


class QueryResponse(BaseModel):
    """Model for query API responses."""
    query: str
    response: str
    sources: List[ChunkResponse]
    similarity_scores: List[float]
    processing_time: float
    token_count: Optional[int] = None
    metadata: Optional[dict] = None


class DocumentUploadResponse(BaseModel):
    """Model for document upload API responses."""
    document_id: str
    filename: str
    status: str
    message: str
    estimated_processing_time: Optional[int] = None
