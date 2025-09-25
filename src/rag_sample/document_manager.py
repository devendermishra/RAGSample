"""
Document management functionality for RAG system.
"""

import os
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import chromadb
from chromadb.config import Settings

from .config import Config
from .web_scraper import WebScraper


class DocumentManager:
    """Manages document loading, processing, and vector store operations."""
    
    def __init__(self, config: Config, chroma_client, collection):
        """Initialize document manager.
        
        Args:
            config: Configuration object
            chroma_client: ChromaDB client
            collection: ChromaDB collection
        """
        self.config = config
        self.chroma_client = chroma_client
        self.collection = collection
        self.web_scraper = WebScraper()
    
    def load_documents(self):
        """Load documents from the documents directory only if not already present in vector DB."""
        documents_path = Path(self.config.documents_path)
        
        if not documents_path.exists():
            print(f"Documents directory {documents_path} not found. Creating it...")
            documents_path.mkdir(parents=True, exist_ok=True)
            return
        
        # Check if vector store already has documents
        if self._vectorstore_has_documents():
            print("Documents already present in vector store. Skipping document loading.")
            return
        
        # Get list of supported files in documents directory
        supported_files = []
        for file_path in documents_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.md']:
                supported_files.append(file_path)
        
        if not supported_files:
            print("No supported documents found in the documents directory.")
            return
        
        print(f"Loading {len(supported_files)} documents from '{documents_path}'...")
        
        # Load all supported document types
        documents = []
        for file_path in supported_files:
            try:
                if file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                    documents.extend(docs)
                elif file_path.suffix.lower() in ['.txt', '.md']:
                    loader = TextLoader(str(file_path))
                    docs = loader.load()
                    documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if documents:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            split_docs = text_splitter.split_documents(documents)
            
            # Add to ChromaDB collection
            self._add_documents_to_chroma(split_docs)
            print(f"Successfully loaded {len(split_docs)} document chunks into vector store.")
        else:
            print("No documents could be loaded.")
    
    def _vectorstore_has_documents(self) -> bool:
        """Check if vector store already contains documents.
        
        Returns:
            True if vector store has documents, False otherwise
        """
        try:
            # Get collection count
            count = self.collection.count()
            return count > 0
        except Exception as e:
            print(f"Error checking vector store contents: {e}")
            return False
    
    def reload_documents(self) -> bool:
        """Force reload all documents from the documents directory.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("Force reloading documents from documents directory...")
            
            # Clear existing collection
            self.collection.delete()  # Delete all documents
            
            # Reload documents
            self.load_documents()
            return True
        except Exception as e:
            print(f"Error reloading documents: {e}")
            return False
    
    def add_document(self, file_path: str) -> bool:
        """Add a new document to the vector store.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                print(f"File not found: {file_path}")
                return False
            
            if not file_path.suffix.lower() in ['.pdf', '.txt', '.md']:
                print(f"Unsupported file type: {file_path.suffix}")
                return False
            
            # Check file size (max 10MB)
            if file_path.stat().st_size > 10 * 1024 * 1024:
                print(f"File too large: {file_path} (max 10MB)")
                return False
            
            # Load document
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
            elif file_path.suffix.lower() in ['.txt', '.md']:
                loader = TextLoader(str(file_path))
                documents = loader.load()
            else:
                print(f"Unsupported file type: {file_path.suffix}")
                return False
            
            if not documents:
                print(f"No content found in {file_path}")
                return False
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            split_docs = text_splitter.split_documents(documents)
            
            # Add to vector store
            self._add_documents_to_chroma(split_docs)
            print(f"Successfully added {len(split_docs)} chunks from {file_path}")
            return True
            
        except Exception as e:
            print(f"Error adding document {file_path}: {e}")
            return False
    
    def add_document_from_url(self, url: str) -> bool:
        """Add a document from a URL.
        
        Args:
            url: URL to scrape content from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Scrape content from URL
            result = self.web_scraper.extract_content(url)
            
            if not result["success"]:
                print(f"Failed to scrape content from {url}: {result['error']}")
                return False
            
            # Create document from scraped content
            metadata = result["metadata"]
            document = Document(
                page_content=result["content"],
                metadata={
                    "source": url,
                    "title": metadata.get("title", ""),
                    "type": "web_page",
                    "domain": metadata.get("domain", ""),
                    "description": metadata.get("description", ""),
                    "author": metadata.get("author", "")
                }
            )
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            split_docs = text_splitter.split_documents([document])
            
            # Add to vector store
            self._add_documents_to_chroma(split_docs)
            print(f"Successfully added {len(split_docs)} chunks from {url}")
            return True
            
        except Exception as e:
            print(f"Error adding document from URL {url}: {e}")
            return False
    
    def remove_document(self, identifier: str) -> bool:
        """Remove a document from the vector store.
        
        Args:
            identifier: Document identifier (filename, URL, or title)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all documents to find matching ones
            results = self.collection.get(include=['metadatas'])
            
            if not results['metadatas']:
                print("No documents found in vector store")
                return False
            
            # Find documents to remove
            docs_to_remove = []
            for i, metadata in enumerate(results['metadatas']):
                source = metadata.get('source', '')
                title = metadata.get('title', '')
                
                if (identifier.lower() in source.lower() or 
                    identifier.lower() in title.lower() or
                    identifier.lower() in str(metadata).lower()):
                    docs_to_remove.append(results['ids'][i])
            
            if not docs_to_remove:
                print(f"No documents found matching '{identifier}'")
                return False
            
            # Remove documents
            self.collection.delete(ids=docs_to_remove)
            print(f"Removed {len(docs_to_remove)} documents matching '{identifier}'")
            return True
            
        except Exception as e:
            print(f"Error removing document '{identifier}': {e}")
            return False
    
    def list_documents(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all documents in the vector store.
        
        Returns:
            Dictionary with documents grouped by source
        """
        try:
            results = self.collection.get(include=['metadatas'])
            
            if not results['metadatas']:
                return {}
            
            # Group documents by source
            documents_by_source = {}
            for metadata in results['metadatas']:
                source = metadata.get('source', 'Unknown')
                if source not in documents_by_source:
                    documents_by_source[source] = []
                
                documents_by_source[source].append({
                    'title': metadata.get('title', 'No title'),
                    'type': metadata.get('type', 'Unknown'),
                    'domain': metadata.get('domain', ''),
                    'author': metadata.get('author', ''),
                    'description': metadata.get('description', '')
                })
            
            return documents_by_source
            
        except Exception as e:
            print(f"Error listing documents: {e}")
            return {}
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about documents in the vector store.
        
        Returns:
            Dictionary with document statistics
        """
        try:
            count = self.collection.count()
            results = self.collection.get(include=['metadatas'])
            
            if not results['metadatas']:
                return {
                    'total_documents': 0,
                    'sources': [],
                    'types': []
                }
            
            # Analyze sources and types
            sources = set()
            types = set()
            
            for metadata in results['metadatas']:
                sources.add(metadata.get('source', 'Unknown'))
                types.add(metadata.get('type', 'Unknown'))
            
            return {
                'total_documents': count,
                'sources': list(sources),
                'types': list(types),
                'source_count': len(sources),
                'type_count': len(types)
            }
            
        except Exception as e:
            print(f"Error getting document stats: {e}")
            return {'total_documents': 0, 'sources': [], 'types': []}
    
    def _add_documents_to_chroma(self, documents: List[Document]):
        """Add documents to ChromaDB collection."""
        if not documents:
            return
        
        # Prepare documents for ChromaDB
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate unique IDs using content hash + timestamp
        ids = []
        for doc in documents:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
            timestamp = int(time.time() * 1000)
            ids.append(f"doc_{content_hash}_{timestamp}")
        
        # Add to collection
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
