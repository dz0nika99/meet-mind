from app.core.database import init_db
from app.core.vector_db import init_vector_collections, close_vector_client
from app.core.logging import get_logger

logger = get_logger(__name__)


async def initialize_services() -> None:
    """Initialize all core services."""
    try:
        logger.info("Initializing core services...")
        
        # Initialize database
        init_db()
        logger.info("Database initialized")
        
        # Initialize vector database
        init_vector_collections()
        logger.info("Vector database initialized")
        
        logger.info("All core services initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize core services", error=str(e))
        raise


async def cleanup_services() -> None:
    """Cleanup all core services."""
    try:
        logger.info("Cleaning up core services...")
        
        # Close vector database client
        close_vector_client()
        logger.info("Vector database client closed")
        
        logger.info("All core services cleaned up successfully")
        
    except Exception as e:
        logger.error("Failed to cleanup core services", error=str(e))
        # Don't raise during cleanup to avoid masking other errors
