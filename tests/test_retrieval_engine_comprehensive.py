"""
Comprehensive tests for retrieval engine functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.rag_sample.retrieval_engine import RetrievalEngine
from src.rag_sample.config import Config
from src.rag_sample.exceptions import RetrievalError


class TestRetrievalEngineComprehensive:
    """Comprehensive test for retrieval engine functionality."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.config = Config()
        self.mock_embeddings = Mock()
        self.mock_collection = Mock()
        self.retrieval_engine = RetrievalEngine(self.config, self.mock_embeddings, self.mock_collection)
    
    def test_retrieval_engine_initialization(self) -> None:
        """Test retrieval engine initialization."""
        assert self.retrieval_engine.config == self.config
        assert self.retrieval_engine.embeddings == self.mock_embeddings
        assert self.retrieval_engine.collection == self.mock_collection
    
    def test_retrieve_documents_success(self) -> None:
        """Test successful document retrieval."""
        # Mock embeddings
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock collection query
        self.mock_collection.query.return_value = {
            'documents': [["Document 1 content", "Document 2 content"]],
            'metadatas': [[{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]],
            'distances': [[0.5, 0.7]]
        }
        
        # Mock document creation
        with patch('src.rag_sample.retrieval_engine.Document') as mock_document:
            mock_doc1 = Mock()
            mock_doc1.page_content = "Document 1 content"
            mock_doc1.metadata = {"source": "doc1.pdf"}
            mock_doc2 = Mock()
            mock_doc2.page_content = "Document 2 content"
            mock_doc2.metadata = {"source": "doc2.pdf"}
            mock_document.side_effect = [mock_doc1, mock_doc2]
            
            result = self.retrieval_engine.retrieve_documents("test query")
            assert len(result) == 2
            assert result[0].page_content == "Document 1 content"
            assert result[1].page_content == "Document 2 content"
    
    def test_retrieve_documents_no_results(self) -> None:
        """Test document retrieval with no results."""
        # Mock embeddings
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock collection query with no results
        self.mock_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        result = self.retrieval_engine.retrieve_documents("test query")
        assert len(result) == 0
    
    def test_retrieve_documents_embedding_error(self) -> None:
        """Test document retrieval when embedding fails."""
        # Mock embeddings to raise error
        self.mock_embeddings.embed_query.side_effect = Exception("Embedding error")
        
        with pytest.raises(RetrievalError):
            self.retrieval_engine.retrieve_documents("test query")
    
    def test_retrieve_documents_collection_error(self) -> None:
        """Test document retrieval when collection query fails."""
        # Mock embeddings
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock collection query to raise error
        self.mock_collection.query.side_effect = Exception("Collection error")
        
        with pytest.raises(RetrievalError):
            self.retrieval_engine.retrieve_documents("test query")
    
    def test_filter_and_rank_documents(self) -> None:
        """Test document filtering and ranking."""
        # Create mock documents
        doc1 = Mock()
        doc1.page_content = "Relevant content about AI"
        doc1.metadata = {"source": "ai.pdf"}
        
        doc2 = Mock()
        doc2.page_content = "Irrelevant content about cooking"
        doc2.metadata = {"source": "cooking.pdf"}
        
        doc3 = Mock()
        doc3.page_content = "Another relevant content about machine learning"
        doc3.metadata = {"source": "ml.pdf"}
        
        documents = [doc1, doc2, doc3]
        distances = [0.3, 0.8, 0.4]
        
        # Mock the relevance check
        with patch.object(self.retrieval_engine, '_is_content_relevant') as mock_relevant:
            mock_relevant.side_effect = [True, False, True]
            
            result = self.retrieval_engine._filter_and_rank_documents(documents, distances, "AI query")
            assert len(result) == 2
            assert result[0] == doc1  # Lower distance, more relevant
            assert result[1] == doc3  # Higher distance, but still relevant
    
    def test_is_content_relevant(self) -> None:
        """Test content relevance checking."""
        # Test relevant content
        relevant_doc = Mock()
        relevant_doc.page_content = "This document is about artificial intelligence and machine learning"
        relevant_doc.metadata = {"title": "AI and ML Guide"}
        
        result = self.retrieval_engine._is_content_relevant(relevant_doc, "artificial intelligence")
        assert result is True
        
        # Test irrelevant content
        irrelevant_doc = Mock()
        irrelevant_doc.page_content = "This document is about cooking recipes and food preparation"
        irrelevant_doc.metadata = {"title": "Cooking Guide"}
        
        result = self.retrieval_engine._is_content_relevant(irrelevant_doc, "artificial intelligence")
        assert result is False
    
    def test_is_content_relevant_with_title(self) -> None:
        """Test content relevance checking with title."""
        # Test relevant title
        relevant_doc = Mock()
        relevant_doc.page_content = "Short content"
        relevant_doc.metadata = {"title": "Artificial Intelligence Guide"}
        
        result = self.retrieval_engine._is_content_relevant(relevant_doc, "artificial intelligence")
        assert result is True
        
        # Test irrelevant title
        irrelevant_doc = Mock()
        irrelevant_doc.page_content = "Short content"
        irrelevant_doc.metadata = {"title": "Cooking Recipes"}
        
        result = self.retrieval_engine._is_content_relevant(irrelevant_doc, "artificial intelligence")
        assert result is False
    
    def test_is_content_relevant_empty_content(self) -> None:
        """Test content relevance checking with empty content."""
        empty_doc = Mock()
        empty_doc.page_content = ""
        empty_doc.metadata = {}
        
        result = self.retrieval_engine._is_content_relevant(empty_doc, "test query")
        assert result is False
    
    def test_is_content_relevant_none_metadata(self) -> None:
        """Test content relevance checking with None metadata."""
        doc = Mock()
        doc.page_content = "Test content"
        doc.metadata = None
        
        result = self.retrieval_engine._is_content_relevant(doc, "test query")
        assert result is True  # Should still work with None metadata
    
    def test_retrieve_documents_with_custom_k(self) -> None:
        """Test document retrieval with custom k value."""
        # Mock embeddings
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock collection query
        self.mock_collection.query.return_value = {
            'documents': [["Document 1 content", "Document 2 content", "Document 3 content"]],
            'metadatas': [[{"source": "doc1.pdf"}, {"source": "doc2.pdf"}, {"source": "doc3.pdf"}]],
            'distances': [[0.5, 0.7, 0.9]]
        }
        
        # Mock document creation
        with patch('src.rag_sample.retrieval_engine.Document') as mock_document:
            mock_docs = [Mock() for _ in range(3)]
            for i, mock_doc in enumerate(mock_docs):
                mock_doc.page_content = f"Document {i+1} content"
                mock_doc.metadata = {"source": f"doc{i+1}.pdf"}
            mock_document.side_effect = mock_docs
            
            result = self.retrieval_engine.retrieve_documents("test query", k=2)
            assert len(result) == 2
    
    def test_retrieve_documents_with_custom_threshold(self) -> None:
        """Test document retrieval with custom threshold."""
        # Mock embeddings
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock collection query
        self.mock_collection.query.return_value = {
            'documents': [["Document 1 content", "Document 2 content"]],
            'metadatas': [[{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]],
            'distances': [[0.3, 0.8]]
        }
        
        # Mock document creation
        with patch('src.rag_sample.retrieval_engine.Document') as mock_document:
            mock_doc1 = Mock()
            mock_doc1.page_content = "Document 1 content"
            mock_doc1.metadata = {"source": "doc1.pdf"}
            mock_doc2 = Mock()
            mock_doc2.page_content = "Document 2 content"
            mock_doc2.metadata = {"source": "doc2.pdf"}
            mock_document.side_effect = [mock_doc1, mock_doc2]
            
            # Mock relevance check
            with patch.object(self.retrieval_engine, '_is_content_relevant', return_value=True):
                result = self.retrieval_engine.retrieve_documents("test query", threshold=0.5)
                assert len(result) == 1  # Only doc1 should pass (distance 0.3 < 0.5)
    
    def test_retrieve_documents_with_debug(self) -> None:
        """Test document retrieval with debug mode."""
        # Enable debug mode
        self.config.enable_retrieval_debug = True
        
        # Mock embeddings
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock collection query
        self.mock_collection.query.return_value = {
            'documents': [["Document 1 content"]],
            'metadatas': [[{"source": "doc1.pdf"}]],
            'distances': [[0.5]]
        }
        
        # Mock document creation
        with patch('src.rag_sample.retrieval_engine.Document') as mock_document:
            mock_doc = Mock()
            mock_doc.page_content = "Document 1 content"
            mock_doc.metadata = {"source": "doc1.pdf"}
            mock_document.return_value = mock_doc
            
            with patch('builtins.print') as mock_print:
                result = self.retrieval_engine.retrieve_documents("test query")
                assert len(result) == 1
                mock_print.assert_called()  # Debug output should be printed
