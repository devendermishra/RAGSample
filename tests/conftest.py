"""
Pytest configuration and fixtures for RAG Sample tests.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Generator

from src.rag_sample.config import Config
from src.rag_sample.rag_engine import RAGEngine
from src.rag_sample.document_manager import DocumentManager
from src.rag_sample.retrieval_engine import RetrievalEngine
from src.rag_sample.web_scraper import WebScraper


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_config() -> Config:
    """Create a mock configuration for testing."""
    config = Config()
    config.groq_api_key = "test-api-key"
    config.vector_db_path = "./test_vector_db"
    config.documents_path = "./test_documents"
    return config


@pytest.fixture
def mock_chroma_client():
    """Create a mock ChromaDB client."""
    client = Mock()
    collection = Mock()
    collection.count.return_value = 0
    collection.get.return_value = {'metadatas': []}
    collection.add.return_value = None
    collection.query.return_value = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    client.get_or_create_collection.return_value = collection
    return client, collection


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings model."""
    embeddings = Mock()
    embeddings.embed_query.return_value = [0.1] * 384  # Mock embedding vector
    embeddings.embed_documents.return_value = [[0.1] * 384]
    return embeddings


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = Mock()
    response = Mock()
    response.content = "Test response"
    llm.invoke.return_value = response
    return llm


@pytest.fixture
def rag_engine(mock_config, mock_chroma_client, mock_embeddings, mock_llm):
    """Create a RAG engine for testing."""
    client, collection = mock_chroma_client
    
    with patch('src.rag_sample.rag_engine.chromadb.PersistentClient', return_value=client):
        with patch('src.rag_sample.rag_engine.HuggingFaceEmbeddings', return_value=mock_embeddings):
            with patch('src.rag_sample.rag_engine.ChatGroq', return_value=mock_llm):
                engine = RAGEngine(config=mock_config)
                engine.collection = collection
                return engine


@pytest.fixture
def document_manager(mock_config, mock_chroma_client):
    """Create a document manager for testing."""
    client, collection = mock_chroma_client
    return DocumentManager(mock_config, client, collection)


@pytest.fixture
def web_scraper():
    """Create a web scraper for testing."""
    return WebScraper()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ['GROQ_API_KEY'] = 'test-api-key'
    yield
    # Cleanup after test
    if 'GROQ_API_KEY' in os.environ:
        del os.environ['GROQ_API_KEY']


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
