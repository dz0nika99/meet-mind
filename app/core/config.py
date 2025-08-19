import os
from typing import List, Optional

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "MeetMind"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=1, env="WORKERS")
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    ALLOWED_HOSTS: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    CORS_ORIGINS: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./meetmind.db",
        env="DATABASE_URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=10, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # Vector Database (Qdrant)
    QDRANT_HOST: str = Field(default="localhost", env="QDRANT_HOST")
    QDRANT_PORT: int = Field(default=6333, env="QDRANT_PORT")
    QDRANT_API_KEY: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = Field(default="meetmind_vectors", env="QDRANT_COLLECTION_NAME")
    QDRANT_VECTOR_SIZE: int = Field(default=384, env="QDRANT_VECTOR_SIZE")
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    OPENAI_MAX_TOKENS: int = Field(default=1000, env="OPENAI_MAX_TOKENS")
    OPENAI_TEMPERATURE: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    
    # Embeddings
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    EMBEDDING_DEVICE: str = Field(default="cpu", env="EMBEDDING_DEVICE")
    EMBEDDING_BATCH_SIZE: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    
    # RAG Configuration
    RAG_CHUNK_SIZE: int = Field(default=1000, env="RAG_CHUNK_SIZE")
    RAG_CHUNK_OVERLAP: int = Field(default=200, env="RAG_CHUNK_OVERLAP")
    RAG_TOP_K: int = Field(default=5, env="RAG_TOP_K")
    RAG_SIMILARITY_THRESHOLD: float = Field(default=0.7, env="RAG_SIMILARITY_THRESHOLD")
    
    # File Upload
    UPLOAD_DIR: str = Field(default="./uploads", env="UPLOAD_DIR")
    MAX_FILE_SIZE: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=[".txt", ".pdf", ".docx", ".md", ".mp3", ".wav", ".m4a"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    
    # Monitoring
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    PROMETHEUS_ENABLED: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    # Cache
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    CACHE_MAX_SIZE: int = Field(default=1000, env="CACHE_MAX_SIZE")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Transcription
    WHISPER_MODEL: str = Field(default="base", env="WHISPER_MODEL")
    WHISPER_DEVICE: str = Field(default="cpu", env="WHISPER_DEVICE")
    
    @validator("ALLOWED_HOSTS", "CORS_ORIGINS", pre=True)
    def parse_list_strings(cls, v):
        """Parse comma-separated strings into lists."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v
    
    @validator("ALLOWED_FILE_TYPES", pre=True)
    def parse_file_types(cls, v):
        """Parse comma-separated file types into lists."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if not v.startswith(("sqlite://", "postgresql://", "mysql://")):
            raise ValueError("Invalid database URL format")
        return v
    
    @validator("QDRANT_HOST")
    def validate_qdrant_host(cls, v):
        """Validate Qdrant host."""
        if not v:
            raise ValueError("Qdrant host cannot be empty")
        return v
    
    @validator("QDRANT_PORT")
    def validate_qdrant_port(cls, v):
        """Validate Qdrant port."""
        if not 1 <= v <= 65535:
            raise ValueError("Qdrant port must be between 1 and 65535")
        return v
    
    @validator("EMBEDDING_DEVICE")
    def validate_embedding_device(cls, v):
        """Validate embedding device."""
        if v not in ["cpu", "cuda", "mps"]:
            raise ValueError("Embedding device must be cpu, cuda, or mps")
        return v
    
    @validator("WHISPER_DEVICE")
    def validate_whisper_device(cls, v):
        """Validate Whisper device."""
        if v not in ["cpu", "cuda", "mps"]:
            raise ValueError("Whisper device must be cpu, cuda, or mps")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create global settings instance
settings = Settings()

# Environment-specific overrides
if settings.ENVIRONMENT == "production":
    settings.DEBUG = False
    settings.LOG_LEVEL = "WARNING"
elif settings.ENVIRONMENT == "testing":
    settings.DATABASE_URL = "sqlite:///./test.db"
    settings.QDRANT_COLLECTION_NAME = "test_vectors"
    settings.DEBUG = True
