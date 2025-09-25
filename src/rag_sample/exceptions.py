"""
Custom exceptions for RAG Sample application.
"""


class RAGSampleError(Exception):
    """Base exception for RAG Sample application."""
    pass


class ConfigurationError(RAGSampleError):
    """Raised when there's a configuration error."""
    pass


class DocumentError(RAGSampleError):
    """Raised when there's an error with document processing."""
    pass


class RetrievalError(RAGSampleError):
    """Raised when there's an error with document retrieval."""
    pass


class WebScrapingError(RAGSampleError):
    """Raised when there's an error with web scraping."""
    pass


class ConversationMemoryError(RAGSampleError):
    """Raised when there's an error with conversation memory."""
    pass


class PromptError(RAGSampleError):
    """Raised when there's an error with prompt building."""
    pass


class VectorStoreError(RAGSampleError):
    """Raised when there's an error with vector store operations."""
    pass


class LLMError(RAGSampleError):
    """Raised when there's an error with LLM operations."""
    pass