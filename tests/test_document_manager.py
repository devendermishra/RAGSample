"""
Tests for document management functionality.
"""

import tempfile
import os
from pathlib import Path
import pytest
from unittest.mock import Mock, patch
from src.rag_sample.document_manager import DocumentManager
from src.rag_sample.config import Config


class TestDocumentManager:
    """Test document manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.chroma_client = Mock()
        self.collection = Mock()
        self.document_manager = DocumentManager(self.config, self.chroma_client, self.collection)
    
    def test_document_manager_initialization(self):
        """Test document manager initialization."""
        assert self.document_manager.config == self.config
        assert self.document_manager.chroma_client == self.chroma_client
        assert self.document_manager.collection == self.collection
    
    def test_vectorstore_has_documents_true(self):
        """Test when vector store has documents."""
        self.collection.count.return_value = 5
        assert self.document_manager._vectorstore_has_documents() is True
    
    def test_vectorstore_has_documents_false(self):
        """Test when vector store has no documents."""
        self.collection.count.return_value = 0
        assert self.document_manager._vectorstore_has_documents() is False
    
    def test_vectorstore_has_documents_error(self):
        """Test when vector store check raises exception."""
        self.collection.count.side_effect = Exception("Database error")
        assert self.document_manager._vectorstore_has_documents() is False
    
    def test_add_document_file_not_found(self):
        """Test adding document when file doesn't exist."""
        result = self.document_manager.add_document("nonexistent.txt")
        assert result is False
    
    def test_add_document_unsupported_type(self):
        """Test adding document with unsupported file type."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b"test content")
            f.flush()
            
            result = self.document_manager.add_document(f.name)
            assert result is False
            
            os.unlink(f.name)
    
    def test_remove_document_no_documents(self):
        """Test removing document when no documents exist."""
        self.collection.get.return_value = {'metadatas': []}
        result = self.document_manager.remove_document("test")
        assert result is False
    
    def test_list_documents_empty(self):
        """Test listing documents when none exist."""
        self.collection.get.return_value = {'metadatas': []}
        result = self.document_manager.list_documents()
        assert result == {}
    
    def test_get_document_stats_empty(self):
        """Test getting document stats when none exist."""
        self.collection.count.return_value = 0
        self.collection.get.return_value = {'metadatas': []}
        result = self.document_manager.get_document_stats()
        assert result['total_documents'] == 0
        assert result['sources'] == []
        assert result['types'] == []
