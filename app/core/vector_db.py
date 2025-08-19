from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Global vector client instance
_vector_client: Optional[QdrantClient] = None


def get_vector_client() -> QdrantClient:
    """Get or create vector database client."""
    global _vector_client
    
    if _vector_client is None:
        try:
            _vector_client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY,
                timeout=30.0
            )
            
            # Test connection
            _vector_client.get_collections()
            logger.info(
                "Vector database client initialized",
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize vector database client",
                error=str(e),
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT
            )
            raise
    
    return _vector_client


def init_vector_collections() -> None:
    """Initialize vector collections."""
    try:
        client = get_vector_client()
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if settings.QDRANT_COLLECTION_NAME not in collection_names:
            # Create collection
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=settings.QDRANT_VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            logger.info(
                "Vector collection created",
                name=settings.QDRANT_COLLECTION_NAME,
                vector_size=settings.QDRANT_VECTOR_SIZE
            )
        else:
            logger.info(
                "Vector collection already exists",
                name=settings.QDRANT_COLLECTION_NAME
            )
            
    except Exception as e:
        logger.error(
            "Failed to initialize vector collections",
            error=str(e)
        )
        raise


def close_vector_client() -> None:
    """Close vector database client."""
    global _vector_client
    
    if _vector_client is not None:
        try:
            _vector_client.close()
            _vector_client = None
            logger.info("Vector database client closed")
        except Exception as e:
            logger.warning(
                "Error closing vector database client",
                error=str(e)
            )
