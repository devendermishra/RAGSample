"""
Tests for custom exceptions.
"""

import pytest

from src.rag_sample.exceptions import (
    RAGSampleError,
    ConfigurationError,
    DocumentError,
    RetrievalError,
    WebScrapingError,
    ConversationMemoryError,
    PromptError,
    VectorStoreError,
    LLMError
)


class TestExceptions:
    """Test custom exception classes."""
    
    def test_rag_sample_error_inheritance(self) -> None:
        """Test that RAGSampleError inherits from Exception."""
        error = RAGSampleError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
    
    def test_configuration_error_inheritance(self) -> None:
        """Test that ConfigurationError inherits from RAGSampleError."""
        error = ConfigurationError("Config error")
        assert isinstance(error, RAGSampleError)
        assert isinstance(error, Exception)
        assert str(error) == "Config error"
    
    def test_document_error_inheritance(self) -> None:
        """Test that DocumentError inherits from RAGSampleError."""
        error = DocumentError("Document error")
        assert isinstance(error, RAGSampleError)
        assert isinstance(error, Exception)
        assert str(error) == "Document error"
    
    def test_retrieval_error_inheritance(self) -> None:
        """Test that RetrievalError inherits from RAGSampleError."""
        error = RetrievalError("Retrieval error")
        assert isinstance(error, RAGSampleError)
        assert isinstance(error, Exception)
        assert str(error) == "Retrieval error"
    
    def test_web_scraping_error_inheritance(self) -> None:
        """Test that WebScrapingError inherits from RAGSampleError."""
        error = WebScrapingError("Web scraping error")
        assert isinstance(error, RAGSampleError)
        assert isinstance(error, Exception)
        assert str(error) == "Web scraping error"
    
    def test_conversation_memory_error_inheritance(self) -> None:
        """Test that ConversationMemoryError inherits from RAGSampleError."""
        error = ConversationMemoryError("Memory error")
        assert isinstance(error, RAGSampleError)
        assert isinstance(error, Exception)
        assert str(error) == "Memory error"
    
    def test_prompt_error_inheritance(self) -> None:
        """Test that PromptError inherits from RAGSampleError."""
        error = PromptError("Prompt error")
        assert isinstance(error, RAGSampleError)
        assert isinstance(error, Exception)
        assert str(error) == "Prompt error"
    
    def test_vector_store_error_inheritance(self) -> None:
        """Test that VectorStoreError inherits from RAGSampleError."""
        error = VectorStoreError("Vector store error")
        assert isinstance(error, RAGSampleError)
        assert isinstance(error, Exception)
        assert str(error) == "Vector store error"
    
    def test_llm_error_inheritance(self) -> None:
        """Test that LLMError inherits from RAGSampleError."""
        error = LLMError("LLM error")
        assert isinstance(error, RAGSampleError)
        assert isinstance(error, Exception)
        assert str(error) == "LLM error"
    
    def test_exception_chaining(self) -> None:
        """Test exception chaining."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            error = ConfigurationError("Config error") from e
            assert error.__cause__ == e
            assert str(error) == "Config error"
    
    def test_exception_with_details(self) -> None:
        """Test exception with additional details."""
        error = ConfigurationError("Config error", "MISSING_API_KEY")
        assert str(error) == "Config error"
        assert hasattr(error, 'error_code')
