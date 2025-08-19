import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from app.services.rag_service import RAGService, RetrievalResult, RAGResponse
from app.models.document import QueryRequest, DocumentChunk
from app.core.config import settings


class TestRAGService:
    """Test suite for RAGService."""
    
    @pytest.fixture
    def rag_service(self):
        """Create a RAG service instance for testing."""
        return RAGService()
    
    @pytest.fixture
    def mock_chunk(self):
        """Create a mock document chunk."""
        chunk = Mock(spec=DocumentChunk)
        chunk.id = "test-chunk-1"
        chunk.document_id = "test-doc-1"
        chunk.content = "This is a test document chunk with some content."
        chunk.chunk_index = 0
        chunk.content_length = 50
        chunk.embedding_model = "test-model"
        chunk.summary = "Test summary"
        chunk.keywords = ["test", "document", "chunk"]
        chunk.created_at = None
        return chunk
    
    @pytest.fixture
    def mock_retrieval_result(self, mock_chunk):
        """Create a mock retrieval result."""
        return RetrievalResult(
            chunk=mock_chunk,
            similarity_score=0.85,
            metadata={"source": "test"}
        )
    
    @pytest.fixture
    def query_request(self):
        """Create a test query request."""
        return QueryRequest(
            query="What is machine learning?",
            query_type="rag",
            top_k=5,
            similarity_threshold=0.7
        )
    
    @pytest.mark.asyncio
    async def test_rag_service_initialization(self, rag_service):
        """Test RAG service initialization."""
        assert rag_service is not None
        assert hasattr(rag_service, 'embedding_service')
        assert hasattr(rag_service, 'vector_service')
        assert hasattr(rag_service, 'llm_service')
        assert hasattr(rag_service, 'retrieval_stats')
    
    @pytest.mark.asyncio
    async def test_query_processing_success(self, rag_service, query_request):
        """Test successful query processing."""
        # Mock dependencies
        with patch.object(rag_service, '_preprocess_query') as mock_preprocess, \
             patch.object(rag_service, '_retrieve_documents') as mock_retrieve, \
             patch.object(rag_service, '_rerank_results') as mock_rerank, \
             patch.object(rag_service, '_generate_answer') as mock_generate, \
             patch.object(rag_service, '_log_query') as mock_log, \
             patch.object(rag_service, '_update_stats') as mock_update:
            
            # Setup mocks
            mock_preprocess.return_value = "processed query"
            mock_retrieve.return_value = [Mock(spec=RetrievalResult)]
            mock_rerank.return_value = [Mock(spec=RetrievalResult)]
            mock_generate.return_value = ("Generated answer", 100)
            
            # Execute query
            result = await rag_service.query(query_request)
            
            # Assertions
            assert isinstance(result, RAGResponse)
            assert result.answer == "Generated answer"
            assert result.token_count == 100
            assert result.processing_time > 0
            
            # Verify mocks were called
            mock_preprocess.assert_called_once_with(query_request.query)
            mock_retrieve.assert_called_once()
            mock_rerank.assert_called_once()
            mock_generate.assert_called_once()
            mock_log.assert_called_once()
            mock_update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_processing_no_results(self, rag_service, query_request):
        """Test query processing when no documents are retrieved."""
        with patch.object(rag_service, '_preprocess_query') as mock_preprocess, \
             patch.object(rag_service, '_retrieve_documents') as mock_retrieve, \
             patch.object(rag_service, '_log_query') as mock_log, \
             patch.object(rag_service, '_update_stats') as mock_update:
            
            # Setup mocks
            mock_preprocess.return_value = "processed query"
            mock_retrieve.return_value = []
            
            # Execute query
            result = await rag_service.query(query_request)
            
            # Assertions
            assert isinstance(result, RAGResponse)
            assert "couldn't find any relevant information" in result.answer
            assert len(result.sources) == 0
            assert result.metadata["retrieval_count"] == 0
    
    @pytest.mark.asyncio
    async def test_query_processing_exception(self, rag_service, query_request):
        """Test query processing when an exception occurs."""
        with patch.object(rag_service, '_preprocess_query') as mock_preprocess:
            # Setup mock to raise exception
            mock_preprocess.side_effect = Exception("Test error")
            
            # Execute query and expect exception
            with pytest.raises(Exception) as exc_info:
                await rag_service.query(query_request)
            
            assert "Test error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_preprocess_query(self, rag_service):
        """Test query preprocessing."""
        query = "  What is   machine learning?  "
        
        with patch.object(rag_service.text_processor, 'clean_text') as mock_clean, \
             patch.object(rag_service.text_processor, 'extract_keywords') as mock_keywords:
            
            # Setup mocks
            mock_clean.return_value = "What is machine learning?"
            mock_keywords.return_value = ["machine", "learning"]
            
            # Execute preprocessing
            result = await rag_service._preprocess_query(query)
            
            # Assertions
            assert "machine learning" in result
            assert "machine" in result
            assert "learning" in result
    
    @pytest.mark.asyncio
    async def test_preprocess_query_exception(self, rag_service):
        """Test query preprocessing when an exception occurs."""
        query = "test query"
        
        with patch.object(rag_service.text_processor, 'clean_text') as mock_clean:
            # Setup mock to raise exception
            mock_clean.side_effect = Exception("Processing error")
            
            # Execute preprocessing
            result = await rag_service._preprocess_query(query)
            
            # Should return original query on error
            assert result == query
    
    @pytest.mark.asyncio
    async def test_retrieve_documents(self, rag_service):
        """Test document retrieval."""
        query = "test query"
        
        with patch.object(rag_service.embedding_service, 'get_embedding') as mock_embedding, \
             patch.object(rag_service.vector_service, 'search') as mock_search, \
             patch.object(rag_service, '_get_chunk_by_id') as mock_get_chunk:
            
            # Setup mocks
            mock_embedding.return_value = [0.1, 0.2, 0.3]
            mock_search.return_value = [
                Mock(chunk_id="chunk-1", similarity_score=0.9, metadata={}),
                Mock(chunk_id="chunk-2", similarity_score=0.8, metadata={})
            ]
            
            mock_chunk1 = Mock(spec=DocumentChunk)
            mock_chunk1.id = "chunk-1"
            mock_chunk2 = Mock(spec=DocumentChunk)
            mock_chunk2.id = "chunk-2"
            
            mock_get_chunk.side_effect = [mock_chunk1, mock_chunk2]
            
            # Execute retrieval
            results = await rag_service._retrieve_documents(query, top_k=2)
            
            # Assertions
            assert len(results) == 2
            assert all(isinstance(r, RetrievalResult) for r in results)
            assert results[0].similarity_score == 0.9
            assert results[1].similarity_score == 0.8
    
    @pytest.mark.asyncio
    async def test_retrieve_documents_exception(self, rag_service):
        """Test document retrieval when an exception occurs."""
        query = "test query"
        
        with patch.object(rag_service.embedding_service, 'get_embedding') as mock_embedding:
            # Setup mock to raise exception
            mock_embedding.side_effect = Exception("Embedding error")
            
            # Execute retrieval
            results = await rag_service._retrieve_documents(query)
            
            # Should return empty list on error
            assert results == []
    
    def test_rerank_results(self, rag_service):
        """Test result reranking."""
        # Create mock results
        result1 = Mock(spec=RetrievalResult)
        result1.similarity_score = 0.7
        result1.chunk.content = "content about machine learning"
        
        result2 = Mock(spec=RetrievalResult)
        result2.similarity_score = 0.8
        result2.chunk.content = "different content about AI"
        
        results = [result1, result2]
        query = "machine learning"
        
        with patch.object(rag_service, '_calculate_content_relevance') as mock_relevance, \
             patch.object(rag_service, '_calculate_source_quality') as mock_quality:
            
            # Setup mocks
            mock_relevance.return_value = 0.9
            mock_quality.return_value = 0.8
            
            # Execute reranking
            reranked = await rag_service._rerank_results(query, results)
            
            # Assertions
            assert len(reranked) == 2
            # Results should be sorted by final score (descending)
            assert reranked[0].similarity_score >= reranked[1].similarity_score
    
    def test_calculate_content_relevance(self, rag_service):
        """Test content relevance calculation."""
        query = "machine learning algorithms"
        content = "This document discusses machine learning algorithms and their applications."
        
        with patch.object(rag_service.text_processor, 'extract_keywords') as mock_keywords:
            # Setup mock
            mock_keywords.side_effect = [
                ["machine", "learning", "algorithms"],  # Query keywords
                ["machine", "learning", "algorithms", "applications"]  # Content keywords
            ]
            
            # Execute calculation
            relevance = rag_service._calculate_content_relevance(query, content)
            
            # Assertions
            assert 0 <= relevance <= 1
            assert relevance > 0.5  # Should have good relevance
    
    def test_calculate_source_quality(self, rag_service, mock_chunk):
        """Test source quality calculation."""
        # Test chunk with good quality indicators
        mock_chunk.content_length = 200
        mock_chunk.summary = "Good summary"
        mock_chunk.keywords = ["keyword1", "keyword2"]
        
        quality = rag_service._calculate_source_quality(mock_chunk)
        
        # Assertions
        assert 0 <= quality <= 1
        assert quality > 0.5  # Should have good quality
    
    def test_apply_diversity_filtering(self, rag_service):
        """Test diversity filtering."""
        # Create similar results
        result1 = Mock(spec=RetrievalResult)
        result1.chunk.content = "content about machine learning"
        
        result2 = Mock(spec=RetrievalResult)
        result2.chunk.content = "similar content about machine learning"
        
        result3 = Mock(spec=RetrievalResult)
        result3.chunk.content = "different content about AI"
        
        results = [result1, result2, result3]
        
        with patch.object(rag_service, '_calculate_content_similarity') as mock_similarity:
            # Setup mock to simulate high similarity between 1 and 2
            mock_similarity.side_effect = [
                0.9,  # result1 vs result2
                0.3,  # result1 vs result3
                0.3   # result2 vs result3
            ]
            
            # Execute filtering
            diverse = rag_service._apply_diversity_filtering(results)
            
            # Should filter out similar results
            assert len(diverse) <= 2
    
    def test_calculate_content_similarity(self, rag_service):
        """Test content similarity calculation."""
        content1 = "machine learning algorithms"
        content2 = "machine learning applications"
        content3 = "completely different topic"
        
        # Test similar content
        similarity1 = rag_service._calculate_content_similarity(content1, content2)
        assert similarity1 > 0.3  # Should have some similarity
        
        # Test different content
        similarity2 = rag_service._calculate_content_similarity(content1, content3)
        assert similarity2 < 0.3  # Should have low similarity
    
    @pytest.mark.asyncio
    async def test_generate_answer(self, rag_service):
        """Test answer generation."""
        query = "What is machine learning?"
        sources = [Mock(spec=RetrievalResult)]
        sources[0].chunk.content = "Machine learning is a subset of AI."
        
        with patch.object(rag_service, '_prepare_context') as mock_context, \
             patch.object(rag_service, '_create_rag_prompt') as mock_prompt, \
             patch.object(rag_service.llm_service, 'generate') as mock_generate:
            
            # Setup mocks
            mock_context.return_value = "Context: Machine learning is a subset of AI."
            mock_prompt.return_value = "Prompt: Answer the question based on context."
            mock_generate.return_value = "Machine learning is a subset of artificial intelligence."
            
            # Execute generation
            answer, token_count = await rag_service._generate_answer(query, sources, "rag")
            
            # Assertions
            assert "Machine learning" in answer
            assert token_count is not None
            assert token_count > 0
    
    @pytest.mark.asyncio
    async def test_generate_answer_no_sources(self, rag_service):
        """Test answer generation when no sources are provided."""
        query = "What is machine learning?"
        sources = []
        
        answer, token_count = await rag_service._generate_answer(query, sources, "rag")
        
        # Should return default message
        assert "couldn't find any relevant information" in answer
        assert token_count is None
    
    def test_prepare_context(self, rag_service):
        """Test context preparation."""
        sources = [
            Mock(spec=RetrievalResult),
            Mock(spec=RetrievalResult)
        ]
        
        sources[0].chunk.content = "Short content."
        sources[1].chunk.content = "This is a very long content that should be truncated because it exceeds the maximum length limit for context preparation in the RAG service."
        
        context = rag_service._prepare_context(sources)
        
        # Assertions
        assert "Source 1:" in context
        assert "Source 2:" in context
        assert "Short content." in context
        assert "..." in context  # Long content should be truncated
    
    def test_create_prompts(self, rag_service):
        """Test prompt creation for different query types."""
        query = "What is machine learning?"
        context = "Context about machine learning."
        
        # Test RAG prompt
        rag_prompt = rag_service._create_rag_prompt(query, context)
        assert "You are a helpful AI assistant" in rag_prompt
        assert query in rag_prompt
        assert context in rag_prompt
        
        # Test summary prompt
        summary_prompt = rag_service._create_summary_prompt(query, context)
        assert "comprehensive summary" in summary_prompt
        assert query in summary_prompt
        assert context in summary_prompt
        
        # Test general prompt
        general_prompt = rag_service._create_general_prompt(query, context)
        assert "Based on the following context" in general_prompt
        assert query in general_prompt
        assert context in general_prompt
    
    @pytest.mark.asyncio
    async def test_get_stats(self, rag_service):
        """Test statistics retrieval."""
        stats = await rag_service.get_stats()
        
        # Assertions
        assert "total_queries" in stats
        assert "average_retrieval_time" in stats
        assert "cache_hit_rate" in stats
        assert "embedding_models" in stats
        assert "active_model" in stats
    
    @pytest.mark.asyncio
    async def test_health_check(self, rag_service):
        """Test health check functionality."""
        with patch.object(rag_service.vector_service, 'health_check') as mock_vector_health, \
             patch.object(rag_service.llm_service, 'health_check') as mock_llm_health:
            
            # Setup mocks
            mock_vector_health.return_value = True
            mock_llm_health.return_value = True
            
            # Execute health check
            health = await rag_service.health_check()
            
            # Assertions
            assert "status" in health
            assert "components" in health
            assert "stats" in health
            assert health["components"]["vector_service"] is True
            assert health["components"]["llm_service"] is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, rag_service):
        """Test health check when components are unhealthy."""
        with patch.object(rag_service.vector_service, 'health_check') as mock_vector_health, \
             patch.object(rag_service.llm_service, 'health_check') as mock_llm_health:
            
            # Setup mocks to simulate failure
            mock_vector_health.return_value = False
            mock_llm_health.return_value = True
            
            # Execute health check
            health = await rag_service.health_check()
            
            # Should be unhealthy
            assert health["status"] == "unhealthy"
            assert health["components"]["vector_service"] is False
            assert health["components"]["llm_service"] is True
    
    def test_update_stats(self, rag_service):
        """Test statistics update."""
        initial_avg = rag_service.retrieval_stats["average_retrieval_time"]
        initial_total = rag_service.retrieval_stats["total_queries"]
        
        # Update stats
        rag_service._update_stats(0.5)
        
        # Assertions
        assert rag_service.retrieval_stats["total_queries"] == initial_total + 1
        assert rag_service.retrieval_stats["average_retrieval_time"] != initial_avg
    
    def test_get_active_embedding_model(self, rag_service):
        """Test active embedding model retrieval."""
        # Test with sentence-transformers
        with patch.dict(rag_service.embedding_models, {
            "sentence-transformers": {"model": "test-model"}
        }):
            model = rag_service._get_active_embedding_model()
            assert "sentence-transformers" in model
        
        # Test with OpenAI
        with patch.dict(rag_service.embedding_models, {
            "openai": {"model": "text-embedding-ada-002"}
        }):
            model = rag_service._get_active_embedding_model()
            assert "openai" in model
        
        # Test with no models
        with patch.dict(rag_service.embedding_models, {}):
            model = rag_service._get_active_embedding_model()
            assert model == "unknown"


class TestRetrievalResult:
    """Test suite for RetrievalResult dataclass."""
    
    def test_retrieval_result_creation(self, mock_chunk):
        """Test RetrievalResult creation."""
        result = RetrievalResult(
            chunk=mock_chunk,
            similarity_score=0.85,
            metadata={"source": "test"}
        )
        
        assert result.chunk == mock_chunk
        assert result.similarity_score == 0.85
        assert result.metadata["source"] == "test"


class TestRAGResponse:
    """Test suite for RAGResponse dataclass."""
    
    def test_rag_response_creation(self, mock_retrieval_result):
        """Test RAGResponse creation."""
        response = RAGResponse(
            answer="Test answer",
            sources=[mock_retrieval_result],
            processing_time=0.5,
            token_count=50,
            metadata={"test": "data"}
        )
        
        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.processing_time == 0.5
        assert response.token_count == 50
        assert response.metadata["test"] == "data"


# Integration tests
@pytest.mark.integration
class TestRAGServiceIntegration:
    """Integration tests for RAG service."""
    
    @pytest.mark.asyncio
    async def test_full_rag_pipeline(self):
        """Test the complete RAG pipeline with real services."""
        # This would test with actual database and vector store
        # Implementation depends on test environment setup
        pass
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self):
        """Test actual embedding generation."""
        # This would test with real embedding models
        # Implementation depends on test environment setup
        pass


# Performance tests
@pytest.mark.performance
class TestRAGServicePerformance:
    """Performance tests for RAG service."""
    
    @pytest.mark.asyncio
    async def test_query_performance(self, rag_service, query_request):
        """Test query processing performance."""
        import time
        
        start_time = time.time()
        result = await rag_service.query(query_request)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Performance assertions
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert result.processing_time < 5.0
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, rag_service):
        """Test batch processing performance."""
        queries = [f"Query {i}" for i in range(10)]
        
        import time
        start_time = time.time()
        
        # Process queries in parallel
        tasks = [rag_service.query(QueryRequest(query=q)) for q in queries]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert total_time < 10.0  # Should complete within 10 seconds
        assert len(results) == 10
