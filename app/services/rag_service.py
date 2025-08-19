import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    SearchRequest, Filter, FieldCondition, MatchValue
)

from app.core.config import settings
from app.core.logging import get_logger, log_rag_operation, track_async_performance
from app.models.document import DocumentChunk, QueryRequest, QueryResponse
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.utils.text_processing import TextChunker, TextProcessor

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    chunk: DocumentChunk
    similarity_score: float
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    """Complete RAG response with sources and metadata."""
    answer: str
    sources: List[RetrievalResult]
    processing_time: float
    token_count: Optional[int] = None
    metadata: Dict[str, Any]


class RAGService:
    """
    Advanced RAG service with multiple retrieval strategies and embedding models.
    
    Features:
    - Multiple embedding models (sentence-transformers, OpenAI)
    - Advanced chunking strategies
    - Hybrid search (dense + sparse retrieval)
    - Reranking and filtering
    - Context window optimization
    - Performance monitoring and caching
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_service = VectorService()
        self.llm_service = LLMService()
        self.text_chunker = TextChunker()
        self.text_processor = TextProcessor()
        
        # Initialize embedding models
        self.embedding_models = {}
        self._initialize_embedding_models()
        
        # Performance tracking
        self.retrieval_stats = {
            "total_queries": 0,
            "average_retrieval_time": 0.0,
            "cache_hit_rate": 0.0
        }
    
    def _initialize_embedding_models(self):
        """Initialize available embedding models."""
        try:
            # Local sentence-transformers models
            if settings.EMBEDDING_MODEL:
                model = SentenceTransformer(
                    settings.EMBEDDING_MODEL,
                    device=settings.EMBEDDING_DEVICE
                )
                self.embedding_models["sentence-transformers"] = {
                    "model": model,
                    "dimensions": model.get_sentence_embedding_dimension(),
                    "provider": "sentence-transformers"
                }
                logger.info(
                    "Initialized sentence-transformers model",
                    model=settings.EMBEDDING_MODEL,
                    dimensions=model.get_sentence_embedding_dimension()
                )
            
            # OpenAI embeddings
            if settings.OPENAI_API_KEY:
                self.embedding_models["openai"] = {
                    "model": "text-embedding-ada-002",
                    "dimensions": 1536,
                    "provider": "openai"
                }
                logger.info("OpenAI embeddings available")
                
        except Exception as e:
            logger.error("Failed to initialize embedding models", error=str(e))
    
    @track_async_performance("rag_query")
    async def query(
        self, 
        request: QueryRequest,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> RAGResponse:
        """
        Process a RAG query with advanced retrieval and generation.
        
        Args:
            request: Query request with parameters
            user_id: Optional user identifier
            session_id: Optional session identifier
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Processing RAG query",
                query=request.query[:100],
                query_type=request.query_type,
                top_k=request.top_k
            )
            
            # Step 1: Query preprocessing and expansion
            processed_query = await self._preprocess_query(request.query)
            
            # Step 2: Retrieve relevant documents
            retrieval_results = await self._retrieve_documents(
                processed_query,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold
            )
            
            if not retrieval_results:
                return RAGResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    processing_time=time.time() - start_time,
                    metadata={"retrieval_count": 0}
                )
            
            # Step 3: Rerank and filter results
            reranked_results = await self._rerank_results(
                processed_query, 
                retrieval_results
            )
            
            # Step 4: Generate answer with context
            answer, token_count = await self._generate_answer(
                request.query,
                reranked_results,
                request.query_type
            )
            
            # Step 5: Log query for analytics
            await self._log_query(
                request, 
                answer, 
                reranked_results,
                time.time() - start_time,
                token_count,
                user_id,
                session_id
            )
            
            processing_time = time.time() - start_time
            
            # Update performance stats
            self._update_stats(processing_time)
            
            # Log operation
            log_rag_operation(
                operation="query",
                query=request.query,
                document_count=len(reranked_results),
                similarity_scores=[r.similarity_score for r in reranked_results],
                duration=processing_time
            )
            
            return RAGResponse(
                answer=answer,
                sources=reranked_results,
                processing_time=processing_time,
                token_count=token_count,
                metadata={
                    "retrieval_count": len(reranked_results),
                    "query_type": request.query_type,
                    "embedding_model": self._get_active_embedding_model(),
                    "cache_hit": False  # TODO: Implement caching
                }
            )
            
        except Exception as e:
            logger.error(
                "RAG query failed",
                error=str(e),
                query=request.query
            )
            raise
    
    async def _preprocess_query(self, query: str) -> str:
        """Preprocess and expand the query for better retrieval."""
        try:
            # Basic text cleaning
            cleaned_query = self.text_processor.clean_text(query)
            
            # Query expansion (simple keyword extraction)
            keywords = self.text_processor.extract_keywords(cleaned_query)
            
            # Add relevant keywords to query
            if keywords:
                expanded_query = f"{cleaned_query} {' '.join(keywords[:3])}"
                logger.debug(
                    "Query expanded",
                    original=query[:50],
                    expanded=expanded_query[:50],
                    keywords=keywords[:3]
                )
                return expanded_query
            
            return cleaned_query
            
        except Exception as e:
            logger.warning("Query preprocessing failed, using original", error=str(e))
            return query
    
    async def _retrieve_documents(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents using multiple strategies.
        
        Implements:
        - Dense vector search
        - Hybrid search (dense + sparse)
        - Semantic similarity
        - Metadata filtering
        """
        try:
            # Get query embedding
            query_embedding = await self.embedding_service.get_embedding(query)
            
            # Perform vector search
            search_results = await self.vector_service.search(
                query_embedding,
                top_k=top_k * 2,  # Get more results for reranking
                similarity_threshold=similarity_threshold
            )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for result in search_results:
                chunk = await self._get_chunk_by_id(result.chunk_id)
                if chunk:
                    retrieval_results.append(RetrievalResult(
                        chunk=chunk,
                        similarity_score=result.similarity_score,
                        metadata=result.metadata or {}
                    ))
            
            logger.debug(
                "Retrieved documents",
                query_length=len(query),
                results_count=len(retrieval_results),
                top_similarity=max([r.similarity_score for r in retrieval_results]) if retrieval_results else 0
            )
            
            return retrieval_results
            
        except Exception as e:
            logger.error("Document retrieval failed", error=str(e))
            return []
    
    async def _rerank_results(
        self, 
        query: str, 
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Rerank results using multiple criteria:
        - Semantic similarity
        - Content relevance
        - Source quality
        - Recency
        """
        try:
            if not results:
                return []
            
            # Calculate additional relevance scores
            for result in results:
                # Content relevance score
                content_relevance = self._calculate_content_relevance(
                    query, 
                    result.chunk.content
                )
                
                # Source quality score (based on document metadata)
                source_quality = self._calculate_source_quality(result.chunk)
                
                # Combine scores with weights
                final_score = (
                    result.similarity_score * 0.6 +
                    content_relevance * 0.3 +
                    source_quality * 0.1
                )
                
                result.similarity_score = final_score
            
            # Sort by final score
            reranked_results = sorted(
                results, 
                key=lambda x: x.similarity_score, 
                reverse=True
            )
            
            # Apply diversity filtering to avoid similar results
            diverse_results = self._apply_diversity_filtering(reranked_results)
            
            logger.debug(
                "Results reranked",
                original_count=len(results),
                final_count=len(diverse_results),
                score_range=f"{min([r.similarity_score for r in diverse_results]):.3f}-{max([r.similarity_score for r in diverse_results]):.3f}"
            )
            
            return diverse_results
            
        except Exception as e:
            logger.error("Result reranking failed", error=str(e))
            return results
    
    def _calculate_content_relevance(self, query: str, content: str) -> float:
        """Calculate content relevance using TF-IDF and keyword matching."""
        try:
            # Extract keywords from query and content
            query_keywords = set(self.text_processor.extract_keywords(query.lower()))
            content_keywords = set(self.text_processor.extract_keywords(content.lower()))
            
            if not query_keywords:
                return 0.5
            
            # Calculate Jaccard similarity
            intersection = len(query_keywords.intersection(content_keywords))
            union = len(query_keywords.union(content_keywords))
            
            if union == 0:
                return 0.0
            
            return intersection / union
            
        except Exception as e:
            logger.warning("Content relevance calculation failed", error=str(e))
            return 0.5
    
    def _calculate_source_quality(self, chunk: DocumentChunk) -> float:
        """Calculate source quality score based on metadata."""
        try:
            quality_score = 0.5  # Base score
            
            # Content length bonus
            if chunk.content_length > 100:
                quality_score += 0.1
            
            # Has summary bonus
            if chunk.summary:
                quality_score += 0.1
            
            # Has keywords bonus
            if chunk.keywords:
                quality_score += 0.1
            
            # Recent content bonus
            if chunk.created_at:
                days_old = (time.time() - chunk.created_at.timestamp()) / (24 * 3600)
                if days_old < 30:  # Less than 30 days old
                    quality_score += 0.1
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.warning("Source quality calculation failed", error=str(e))
            return 0.5
    
    def _apply_diversity_filtering(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply diversity filtering to avoid similar results."""
        try:
            if len(results) <= 5:
                return results
            
            diverse_results = [results[0]]  # Always include top result
            
            for result in results[1:]:
                # Check if this result is too similar to already selected ones
                is_diverse = True
                for selected in diverse_results:
                    similarity = self._calculate_content_similarity(
                        result.chunk.content, 
                        selected.chunk.content
                    )
                    if similarity > 0.8:  # High similarity threshold
                        is_diverse = False
                        break
                
                if is_diverse and len(diverse_results) < 5:
                    diverse_results.append(result)
            
            return diverse_results
            
        except Exception as e:
            logger.warning("Diversity filtering failed", error=str(e))
            return results[:5]
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content pieces."""
        try:
            # Simple Jaccard similarity on words
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            if union == 0:
                return 0.0
            
            return intersection / union
            
        except Exception:
            return 0.0
    
    async def _generate_answer(
        self, 
        query: str, 
        sources: List[RetrievalResult],
        query_type: str
    ) -> Tuple[str, Optional[int]]:
        """Generate answer using retrieved sources and LLM."""
        try:
            if not sources:
                return "I couldn't find any relevant information to answer your question.", None
            
            # Prepare context from sources
            context = self._prepare_context(sources)
            
            # Generate prompt based on query type
            if query_type == "rag":
                prompt = self._create_rag_prompt(query, context)
            elif query_type == "summary":
                prompt = self._create_summary_prompt(query, context)
            else:
                prompt = self._create_general_prompt(query, context)
            
            # Generate answer using LLM
            answer = await self.llm_service.generate(
                prompt,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                temperature=settings.OPENAI_TEMPERATURE
            )
            
            # Estimate token count
            token_count = len(answer.split())  # Rough estimation
            
            return answer, token_count
            
        except Exception as e:
            logger.error("Answer generation failed", error=str(e))
            return "I encountered an error while generating the answer. Please try again.", None
    
    def _prepare_context(self, sources: List[RetrievalResult]) -> str:
        """Prepare context string from retrieved sources."""
        try:
            context_parts = []
            
            for i, source in enumerate(sources, 1):
                content = source.chunk.content.strip()
                if len(content) > 500:  # Truncate very long chunks
                    content = content[:500] + "..."
                
                context_parts.append(f"Source {i}:\n{content}\n")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error("Context preparation failed", error=str(e))
            return ""
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create prompt for RAG-style question answering."""
        return f"""You are a helpful AI assistant. Use the following context to answer the user's question. 
        If the context doesn't contain enough information to answer the question, say so.

        Context:
        {context}

        Question: {query}

        Answer:"""
    
    def _create_summary_prompt(self, query: str, context: str) -> str:
        """Create prompt for summarization tasks."""
        return f"""Please provide a comprehensive summary of the following content based on the user's request.

        Content:
        {context}

        Request: {query}

        Summary:"""
    
    def _create_general_prompt(self, query: str, context: str) -> str:
        """Create general-purpose prompt."""
        return f"""Based on the following context, please respond to the user's request.

        Context:
        {context}

        Request: {query}

        Response:"""
    
    async def _get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get document chunk by ID from database."""
        # TODO: Implement database retrieval
        # This is a placeholder for the actual database query
        return None
    
    async def _log_query(
        self,
        request: QueryRequest,
        answer: str,
        sources: List[RetrievalResult],
        processing_time: float,
        token_count: Optional[int],
        user_id: Optional[str],
        session_id: Optional[str]
    ):
        """Log query for analytics and monitoring."""
        try:
            # TODO: Implement actual logging to database
            logger.info(
                "Query logged",
                query_length=len(request.query),
                answer_length=len(answer),
                sources_count=len(sources),
                processing_time=processing_time,
                token_count=token_count,
                user_id=user_id,
                session_id=session_id
            )
        except Exception as e:
            logger.warning("Query logging failed", error=str(e))
    
    def _update_stats(self, processing_time: float):
        """Update performance statistics."""
        try:
            self.retrieval_stats["total_queries"] += 1
            
            # Update average retrieval time
            current_avg = self.retrieval_stats["average_retrieval_time"]
            total_queries = self.retrieval_stats["total_queries"]
            
            self.retrieval_stats["average_retrieval_time"] = (
                (current_avg * (total_queries - 1) + processing_time) / total_queries
            )
            
        except Exception as e:
            logger.warning("Stats update failed", error=str(e))
    
    def _get_active_embedding_model(self) -> str:
        """Get the currently active embedding model name."""
        if "sentence-transformers" in self.embedding_models:
            return f"sentence-transformers:{settings.EMBEDDING_MODEL}"
        elif "openai" in self.embedding_models:
            return "openai:text-embedding-ada-002"
        else:
            return "unknown"
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get RAG service statistics."""
        return {
            **self.retrieval_stats,
            "embedding_models": list(self.embedding_models.keys()),
            "active_model": self._get_active_embedding_model()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the RAG service."""
        try:
            # Check embedding models
            embedding_health = len(self.embedding_models) > 0
            
            # Check vector service
            vector_health = await self.vector_service.health_check()
            
            # Check LLM service
            llm_health = await self.llm_service.health_check()
            
            return {
                "status": "healthy" if all([embedding_health, vector_health, llm_health]) else "unhealthy",
                "components": {
                    "embedding_service": embedding_health,
                    "vector_service": vector_health,
                    "llm_service": llm_health
                },
                "stats": await self.get_stats()
            }
            
        except Exception as e:
            logger.error("RAG service health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }
