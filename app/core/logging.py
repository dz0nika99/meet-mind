import logging
import sys
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory


def setup_logging() -> None:
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def log_correlation_id(correlation_id: str) -> Dict[str, str]:
    """Create a context dict with correlation ID for logging."""
    return {"correlation_id": correlation_id}


def log_performance(operation: str, duration: float, **kwargs) -> None:
    """Log performance metrics with structured data."""
    logger = get_logger("performance")
    logger.info(
        "Operation completed",
        operation=operation,
        duration_ms=round(duration * 1000, 2),
        **kwargs
    )


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    duration: float,
    correlation_id: str,
    **kwargs
) -> None:
    """Log API request details with structured data."""
    logger = get_logger("api")
    logger.info(
        "API request completed",
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=round(duration * 1000, 2),
        correlation_id=correlation_id,
        **kwargs
    )


def log_error(
    error: Exception,
    context: Dict[str, Any] = None,
    correlation_id: str = None
) -> None:
    """Log errors with structured context."""
    logger = get_logger("error")
    
    log_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error_traceback": getattr(error, "__traceback__", None),
    }
    
    if context:
        log_data.update(context)
    
    if correlation_id:
        log_data["correlation_id"] = correlation_id
    
    logger.error("Error occurred", **log_data)


def log_rag_operation(
    operation: str,
    query: str = None,
    document_count: int = None,
    similarity_scores: list = None,
    duration: float = None,
    **kwargs
) -> None:
    """Log RAG operation details with structured data."""
    logger = get_logger("rag")
    
    log_data = {
        "operation": operation,
        "duration_ms": round(duration * 1000, 2) if duration else None,
    }
    
    if query:
        log_data["query"] = query[:100] + "..." if len(query) > 100 else query
    
    if document_count is not None:
        log_data["document_count"] = document_count
    
    if similarity_scores:
        log_data["similarity_scores"] = [round(score, 4) for score in similarity_scores]
    
    log_data.update(kwargs)
    
    logger.info("RAG operation completed", **log_data)


def log_vector_operation(
    operation: str,
    collection: str,
    vector_count: int = None,
    duration: float = None,
    **kwargs
) -> None:
    """Log vector database operation details."""
    logger = get_logger("vector_db")
    
    log_data = {
        "operation": operation,
        "collection": collection,
        "duration_ms": round(duration * 1000, 2) if duration else None,
    }
    
    if vector_count is not None:
        log_data["vector_count"] = vector_count
    
    log_data.update(kwargs)
    
    logger.info("Vector operation completed", **log_data)


def log_document_processing(
    file_name: str,
    file_size: int,
    file_type: str,
    processing_time: float,
    chunks_created: int,
    **kwargs
) -> None:
    """Log document processing details."""
    logger = get_logger("document_processing")
    
    logger.info(
        "Document processed",
        file_name=file_name,
        file_size_mb=round(file_size / (1024 * 1024), 2),
        file_type=file_type,
        processing_time_ms=round(processing_time * 1000, 2),
        chunks_created=chunks_created,
        **kwargs
    )


def log_embedding_generation(
    model: str,
    text_length: int,
    batch_size: int,
    duration: float,
    device: str,
    **kwargs
) -> None:
    """Log embedding generation details."""
    logger = get_logger("embeddings")
    
    logger.info(
        "Embeddings generated",
        model=model,
        text_length=text_length,
        batch_size=batch_size,
        duration_ms=round(duration * 1000, 2),
        device=device,
        **kwargs
    )


# Performance tracking decorator
def track_performance(operation_name: str):
    """Decorator to track performance of functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                log_performance(operation_name, duration, success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                log_performance(operation_name, duration, success=False, error=str(e))
                raise
        
        return wrapper
    return decorator


# Async performance tracking decorator
def track_async_performance(operation_name: str):
    """Decorator to track performance of async functions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                log_performance(operation_name, duration, success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                log_performance(operation_name, duration, success=False, error=str(e))
                raise
        
        return wrapper
    return decorator
