import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
import structlog

from app.core.config import settings
from app.core.logging import get_logger, log_rag_operation
from app.models.document import QueryRequest, QueryResponse, DocumentResponse
from app.services.rag_service import RAGService, RAGResponse
from app.core.auth import get_current_user
from app.utils.rate_limiting import rate_limit

logger = get_logger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
@rate_limit(max_requests=60, window_seconds=60)
async def query_rag(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Query the RAG system with a natural language question.
    
    This endpoint provides advanced retrieval-augmented generation capabilities:
    - Semantic search across document chunks
    - Multi-model embedding support
    - Intelligent result reranking
    - Context-aware answer generation
    
    Args:
        request: Query request with parameters
        background_tasks: Background task manager
        current_user: Authenticated user (optional)
        
    Returns:
        QueryResponse with answer, sources, and metadata
        
    Raises:
        HTTPException: For various error conditions
    """
    start_time = time.time()
    
    try:
        # Initialize RAG service
        rag_service = RAGService()
        
        # Process query
        rag_response = await rag_service.query(
            request=request,
            user_id=current_user,
            session_id=None  # TODO: Implement session management
        )
        
        # Convert to API response format
        sources = []
        for source in rag_response.sources:
            sources.append({
                "id": source.chunk.id,
                "document_id": source.chunk.document_id,
                "content": source.chunk.content,
                "chunk_index": source.chunk.chunk_index,
                "summary": source.chunk.summary,
                "keywords": source.chunk.keywords,
                "content_length": source.chunk.content_length,
                "embedding_model": source.chunk.embedding_model,
                "created_at": source.chunk.created_at
            })
        
        # Log operation
        log_rag_operation(
            operation="api_query",
            query=request.query,
            document_count=len(rag_response.sources),
            similarity_scores=[s.similarity_score for s in rag_response.sources],
            duration=rag_response.processing_time
        )
        
        # Add background task for analytics
        background_tasks.add_task(
            _log_query_analytics,
            request.query,
            rag_response.answer,
            len(rag_response.sources),
            rag_response.processing_time,
            current_user
        )
        
        return QueryResponse(
            query=request.query,
            response=rag_response.answer,
            sources=sources,
            similarity_scores=[s.similarity_score for s in rag_response.sources],
            processing_time=rag_response.processing_time,
            token_count=rag_response.token_count,
            metadata=rag_response.metadata
        )
        
    except Exception as e:
        logger.error(
            "RAG query failed",
            error=str(e),
            query=request.query,
            user_id=current_user
        )
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@router.post("/chat")
async def chat_with_rag(
    message: str = Query(..., description="Chat message"),
    conversation_id: Optional[str] = Query(None, description="Conversation ID"),
    top_k: int = Query(5, description="Number of top results to return"),
    similarity_threshold: float = Query(0.7, description="Similarity threshold"),
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Interactive chat endpoint with RAG capabilities.
    
    Provides conversational AI with context from your knowledge base.
    Maintains conversation history for better context understanding.
    
    Args:
        message: User's chat message
        conversation_id: Optional conversation identifier
        top_k: Number of relevant sources to retrieve
        similarity_threshold: Minimum similarity for sources
        current_user: Authenticated user (optional)
        
    Returns:
        Streaming response with chat completion
    """
    try:
        # Create query request
        request = QueryRequest(
            query=message,
            query_type="chat",
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Initialize RAG service
        rag_service = RAGService()
        
        # Process query
        rag_response = await rag_service.query(
            request=request,
            user_id=current_user,
            session_id=conversation_id
        )
        
        # Return streaming response
        return StreamingResponse(
            _stream_chat_response(rag_response.answer),
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(
            "Chat query failed",
            error=str(e),
            message=message,
            user_id=current_user
        )
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )


@router.post("/summarize")
async def summarize_documents(
    query: str = Query(..., description="Summarization request"),
    document_ids: List[str] = Query(..., description="Document IDs to summarize"),
    max_length: int = Query(500, description="Maximum summary length"),
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Generate summaries of specific documents or document collections.
    
    Uses advanced NLP techniques to create coherent, informative summaries
    based on user requirements and document content.
    
    Args:
        query: User's summarization request
        document_ids: List of document IDs to summarize
        max_length: Maximum length of the summary
        current_user: Authenticated user (optional)
        
    Returns:
        Summary response with generated text
    """
    try:
        # Create query request for summarization
        request = QueryRequest(
            query=query,
            query_type="summary",
            top_k=len(document_ids),
            similarity_threshold=0.5  # Lower threshold for summarization
        )
        
        # Initialize RAG service
        rag_service = RAGService()
        
        # TODO: Implement document-specific retrieval
        # For now, use general query
        rag_response = await rag_service.query(
            request=request,
            user_id=current_user
        )
        
        # Truncate response if needed
        summary = rag_response.answer
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return {
            "summary": summary,
            "document_ids": document_ids,
            "max_length": max_length,
            "actual_length": len(summary),
            "processing_time": rag_response.processing_time
        }
        
    except Exception as e:
        logger.error(
            "Document summarization failed",
            error=str(e),
            query=query,
            document_ids=document_ids,
            user_id=current_user
        )
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}"
        )


@router.get("/sources")
async def get_rag_sources(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(10, description="Number of sources to return"),
    similarity_threshold: float = Query(0.7, description="Similarity threshold"),
    include_content: bool = Query(True, description="Include source content"),
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Retrieve relevant document sources without generating an answer.
    
    Useful for research, fact-checking, or when you want to examine
    the source materials directly.
    
    Args:
        query: Search query
        top_k: Number of sources to return
        similarity_threshold: Minimum similarity score
        include_content: Whether to include source content
        current_user: Authenticated user (optional)
        
    Returns:
        List of relevant document sources
    """
    try:
        # Initialize RAG service
        rag_service = RAGService()
        
        # Create query request
        request = QueryRequest(
            query=query,
            query_type="retrieval_only",
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Get retrieval results only
        rag_response = await rag_service.query(request, current_user)
        
        # Format sources
        sources = []
        for source in rag_response.sources:
            source_data = {
                "id": source.chunk.id,
                "document_id": source.chunk.document_id,
                "chunk_index": source.chunk.chunk_index,
                "similarity_score": source.similarity_score,
                "metadata": source.metadata
            }
            
            if include_content:
                source_data["content"] = source.chunk.content
                source_data["summary"] = source.chunk.summary
                source_data["keywords"] = source.chunk.keywords
            
            sources.append(source_data)
        
        return {
            "query": query,
            "sources": sources,
            "total_count": len(sources),
            "processing_time": rag_response.processing_time
        }
        
    except Exception as e:
        logger.error(
            "Source retrieval failed",
            error=str(e),
            query=query,
            user_id=current_user
        )
        raise HTTPException(
            status_code=500,
            detail=f"Source retrieval failed: {str(e)}"
        )


@router.post("/batch-query")
async def batch_rag_query(
    queries: List[str] = Query(..., description="List of queries to process"),
    query_type: str = Query("rag", description="Type of query processing"),
    top_k: int = Query(5, description="Number of top results per query"),
    similarity_threshold: float = Query(0.7, description="Similarity threshold"),
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Process multiple RAG queries in batch for efficiency.
    
    Optimized for processing multiple related queries or when you need
    to analyze multiple aspects of your knowledge base.
    
    Args:
        queries: List of queries to process
        query_type: Type of query processing
        top_k: Number of top results per query
        similarity_threshold: Minimum similarity threshold
        current_user: Authenticated user (optional)
        
    Returns:
        Batch processing results
    """
    try:
        if len(queries) > 10:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 queries allowed per batch"
            )
        
        # Initialize RAG service
        rag_service = RAGService()
        
        # Process queries in parallel
        results = []
        total_processing_time = 0
        
        for query in queries:
            request = QueryRequest(
                query=query,
                query_type=query_type,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            rag_response = await rag_service.query(request, current_user)
            
            results.append({
                "query": query,
                "answer": rag_response.answer,
                "sources_count": len(rag_response.sources),
                "processing_time": rag_response.processing_time,
                "token_count": rag_response.token_count
            })
            
            total_processing_time += rag_response.processing_time
        
        return {
            "queries_processed": len(queries),
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(queries),
            "results": results
        }
        
    except Exception as e:
        logger.error(
            "Batch query processing failed",
            error=str(e),
            queries_count=len(queries),
            user_id=current_user
        )
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.get("/stats")
async def get_rag_stats(
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Get RAG system statistics and performance metrics.
    
    Provides insights into system usage, performance, and health.
    Useful for monitoring and optimization.
    
    Args:
        current_user: Authenticated user (optional)
        
    Returns:
        RAG system statistics
    """
    try:
        # Initialize RAG service
        rag_service = RAGService()
        
        # Get statistics
        stats = await rag_service.get_stats()
        health = await rag_service.health_check()
        
        return {
            "system_health": health,
            "performance_stats": stats,
            "configuration": {
                "embedding_model": settings.EMBEDDING_MODEL,
                "vector_size": settings.QDRANT_VECTOR_SIZE,
                "chunk_size": settings.RAG_CHUNK_SIZE,
                "chunk_overlap": settings.RAG_CHUNK_OVERLAP,
                "similarity_threshold": settings.RAG_SIMILARITY_THRESHOLD
            }
        }
        
    except Exception as e:
        logger.error("Failed to get RAG stats", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


# Helper functions
async def _log_query_analytics(
    query: str,
    answer: str,
    sources_count: int,
    processing_time: float,
    user_id: Optional[str]
):
    """Background task to log query analytics."""
    try:
        # TODO: Implement analytics logging
        logger.info(
            "Query analytics logged",
            query_length=len(query),
            answer_length=len(answer),
            sources_count=sources_count,
            processing_time=processing_time,
            user_id=user_id
        )
    except Exception as e:
        logger.warning("Analytics logging failed", error=str(e))


def _stream_chat_response(response: str):
    """Stream chat response character by character."""
    for char in response:
        yield char
        time.sleep(0.01)  # Small delay for streaming effect
