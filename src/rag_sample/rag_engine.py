"""
RAG Engine implementation for document chat functionality.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .config import Config
from .conversation_memory import ConversationMemory
from .web_scraper import WebScraper
from .prompt_builder import PromptManager


class RAGEngine:
    """RAG Engine for document-based question answering."""
    
    def __init__(self, model: str = None, temperature: float = None, 
                 max_tokens: int = None, config: Optional[Config] = None):
        """Initialize RAG Engine.
        
        Args:
            model: Groq model to use (uses config if None)
            temperature: Temperature for response generation (uses config if None)
            max_tokens: Maximum tokens for response (uses config if None)
            config: Configuration object
        """
        self.config = config or Config()
        self.model = model or self.config.groq_model
        self.temperature = temperature if temperature is not None else self.config.temperature
        self.max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        
        # Initialize components
        self._setup_embeddings()
        self._setup_llm()
        self._setup_vectorstore()
        self._setup_qa_chain()
        
        # Initialize conversation memory
        self.conversation_memory = ConversationMemory(
            max_tokens=self.config.max_conversation_tokens,
            summarization_threshold=self.config.summarization_threshold,
            llm=self.llm
        ) if self.config.enable_conversation_memory else None
        
        # Initialize web scraper
        self.web_scraper = WebScraper()
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()
    
    def _setup_embeddings(self):
        """Setup embeddings model."""
        # Use HuggingFace embeddings instead of OpenAI for better compatibility
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def _setup_llm(self):
        """Setup language model."""
        self.llm = ChatGroq(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _setup_vectorstore(self):
        """Setup vector store using ChromaDB directly."""
        # Create vector database directory
        os.makedirs(self.config.vector_db_path, exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.vector_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection_name = self.config.vector_db_collection
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "hnsw:batch_size": 10000}
            )
        
        # Load documents if they exist
        self._load_documents()
    
    def _load_documents(self):
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
    
    def _add_documents_to_chroma(self, documents: List[Document]):
        """Add documents to ChromaDB collection."""
        if not documents:
            return
        
        # Prepare data for ChromaDB
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate unique IDs based on content hash and timestamp
        import hashlib
        import time
        ids = []
        for i, doc in enumerate(documents):
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
            timestamp = int(time.time() * 1000)  # milliseconds
            ids.append(f"doc_{content_hash}_{timestamp}_{i}")
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)
        
        # Add to collection
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
    
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
            self._load_documents()
            return True
        except Exception as e:
            print(f"Error reloading documents: {e}")
            return False
    
    def _setup_qa_chain(self):
        """Setup question-answering chain."""
        # We'll handle retrieval directly in the chat method
        # No need for LangChain's RetrievalQA chain
        pass
    
    def chat(self, question: str) -> str:
        """Chat with the RAG system.
        
        Args:
            question: User's question
            
        Returns:
            System's response
        """
        try:
            # Add user question to conversation memory
            if self.conversation_memory:
                self.conversation_memory.add_message("user", question)
            
            # Get conversation context if available
            conversation_context = ""
            if self.conversation_memory:
                conversation_context = self.conversation_memory.get_conversation_context()
            
            # STEP 1: Retrieve relevant documents first
            retrieved_docs = self._retrieve_documents(question)
            
            if self.config.enable_retrieval_debug:
                print(f"\n🔍 Retrieved {len(retrieved_docs)} documents:")
                for i, doc in enumerate(retrieved_docs, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    title = doc.metadata.get('title', 'No title')
                    print(f"  {i}. {title} (from {source})")
                print()
            
            # STEP 2: Build context from retrieved documents
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # STEP 3: Build prompt with retrieved context using Ready Tensor techniques
            if conversation_context:
                # Use conversation context prompt with retrieved documents
                prompt_input = f"""Previous conversation:
{conversation_context}

Current question: {question}

Retrieved documents context:
{context_text}"""
                
                enhanced_question = self.prompt_manager.build_prompt(
                    "conversation_context_prompt",
                    input_data=prompt_input
                )
            else:
                # Use advanced RAG prompt for complex queries
                if len(retrieved_docs) > 2 or len(question.split()) > 10:
                    # Use advanced prompt for complex queries
                    prompt_input = f"""Question: {question}

Retrieved documents context:
{context_text}"""
                    
                    enhanced_question = self.prompt_manager.build_prompt(
                        "advanced_rag_prompt",
                        input_data=prompt_input
                    )
                else:
                    # Use standard RAG prompt for simple queries
                    prompt_input = f"""Question: {question}

Retrieved documents context:
{context_text}"""
                    
                    enhanced_question = self.prompt_manager.build_prompt(
                        "rag_assistant_prompt",
                        input_data=prompt_input
                    )
            
            # STEP 4: Create final prompt for LLM using enhanced question
            final_prompt = f"""Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context_text}

Question: {enhanced_question}

Answer:"""
            
            # STEP 5: Get response from LLM
            response = self.llm.invoke(final_prompt)
            if hasattr(response, 'content'):
                result = {"result": response.content}
            else:
                result = {"result": str(response)}
            
            # Add assistant response to conversation memory
            if self.conversation_memory:
                self.conversation_memory.add_message("assistant", result)
            
            return result.get("result", "RESULT NOT FOUND")
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            # Add error to conversation memory
            if self.conversation_memory:
                self.conversation_memory.add_message("assistant", error_msg)
            return error_msg
    
    def _retrieve_documents(self, question: str) -> List[Document]:
        """Retrieve relevant documents using Ready Tensor advanced techniques.
        
        Args:
            question: User's question
            
        Returns:
            List of relevant documents
        """
        try:
            # Generate embedding for the question
            question_embedding = self.embeddings.embed_query(question)
            
            # Query ChromaDB collection
            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=self.config.retrieval_top_k * 2,  # Get more candidates for filtering
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert results to documents with scores
            docs_with_scores = []
            if results['documents'] and results['documents'][0]:
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    doc = Document(page_content=doc_text, metadata=metadata)
                    docs_with_scores.append((doc, distance))
            
            # Apply Ready Tensor's filtering and ranking
            filtered_docs = self._filter_and_rank_documents(docs_with_scores, question)
            
            return filtered_docs
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
    
    def _filter_and_rank_documents(self, docs_with_scores: List[tuple], question: str) -> List[Document]:
        """Filter and rank documents using Ready Tensor techniques."""
        filtered_docs = []
        
        for doc, score in docs_with_scores:
            # Apply similarity threshold
            if score <= self.config.retrieval_threshold:
                # Apply Ready Tensor's content relevance check
                #if self._is_content_relevant(doc, question):
                filtered_docs.append(doc)
        
        # Limit to top_k results
        return filtered_docs[:self.config.retrieval_top_k]
    
    def _is_content_relevant(self, doc: Document, question: str) -> bool:
        """Check if document content is relevant to the question using Ready Tensor techniques."""
        # Basic relevance checks
        question_lower = question.lower()
        content_lower = doc.page_content.lower()
        
        # Check for key terms overlap
        question_terms = set(question_lower.split())
        content_terms = set(content_lower.split())
        
        # Calculate basic relevance score
        overlap = len(question_terms.intersection(content_terms))
        relevance_score = overlap / len(question_terms) if question_terms else 0
        
        # Apply minimum relevance threshold
        return relevance_score >= 0.1  # 10% term overlap minimum
    
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
                print(f"Error: File not found: {file_path}")
                return False
            
            if not file_path.is_file():
                print(f"Error: Path is not a file: {file_path}")
                return False
            
            # Check file size (limit to 50MB)
            file_size = file_path.stat().st_size
            if file_size > 50 * 1024 * 1024:  # 50MB
                print(f"Error: File too large ({file_size / (1024*1024):.1f}MB). Maximum size is 50MB.")
                return False
            
            # Load document based on file type
            documents = []
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.pdf':
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
            elif file_ext in ['.txt', '.md']:
                loader = TextLoader(str(file_path))
                documents = loader.load()
            else:
                print(f"Error: Unsupported file type '{file_ext}'. Supported types: .pdf, .txt, .md")
                return False
            
            if not documents:
                print(f"Error: No content found in file: {file_path}")
                return False
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            split_docs = text_splitter.split_documents(documents)
            
            if not split_docs:
                print(f"Error: No text chunks created from file: {file_path}")
                return False
            
            # Add to ChromaDB collection
            self._add_documents_to_chroma(split_docs)
            print(f"Successfully added {len(split_docs)} chunks from {file_path.name}")
            return True
            
        except Exception as e:
            print(f"Error adding document '{file_path}': {str(e)}")
            return False
    
    def add_document_from_url(self, url: str) -> bool:
        """Add a document from a URL to the vector store.
        
        Args:
            url: URL to scrape and add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Scraping content from: {url}")
            
            # Extract content from URL
            result = self.web_scraper.extract_content(url)
            
            if not result["success"]:
                print(f"Error scraping URL: {result['error']}")
                return False
            
            content = result["content"]
            metadata = result["metadata"]
            
            if not content.strip():
                print("Error: No content extracted from URL")
                return False
            
            # Create a document object
            doc_metadata = {
                "source": url,
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "author": metadata.get("author", ""),
                "domain": metadata.get("domain", ""),
                "type": "web_page"
            }
            
            document = Document(page_content=content, metadata=doc_metadata)
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            split_docs = text_splitter.split_documents([document])
            
            if not split_docs:
                print("Error: No text chunks created from URL content")
                return False
            
            # Add to ChromaDB collection
            self._add_documents_to_chroma(split_docs)
            print(f"Successfully added {len(split_docs)} chunks from {url}")
            return True
            
        except Exception as e:
            print(f"Error adding document from URL '{url}': {str(e)}")
            return False
    
    def remove_document(self, document_identifier: str) -> bool:
        """Remove a document from the vector store.
        
        Args:
            document_identifier: Document identifier (filename, URL, or source)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all documents from the collection
            results = self.collection.get()
            
            if not results or not results.get('ids'):
                print("No documents found in the vector store.")
                return False
            
            # Find documents to remove based on identifier
            documents_to_remove = []
            metadata_list = results.get('metadatas', [])
            ids_list = results.get('ids', [])
            
            for i, (doc_id, metadata) in enumerate(zip(ids_list, metadata_list)):
                if metadata:
                    # Check various identifier fields
                    source = metadata.get('source', '')
                    title = metadata.get('title', '')
                    
                    # Match by filename, URL, or title
                    if (document_identifier.lower() in source.lower() or
                        document_identifier.lower() in title.lower() or
                        document_identifier.lower() in source.lower()):
                        documents_to_remove.append(doc_id)
            
            if not documents_to_remove:
                print(f"No documents found matching '{document_identifier}'")
                return False
            
            # Remove documents from collection
            self.collection.delete(ids=documents_to_remove)
            print(f"Successfully removed {len(documents_to_remove)} document chunks matching '{document_identifier}'")
            return True
            
        except Exception as e:
            print(f"Error removing document '{document_identifier}': {str(e)}")
            return False
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the vector store.
        
        Returns:
            List of document information
        """
        try:
            results = self.collection.get()
            
            if not results or not results.get('ids'):
                return []
            
            documents = []
            metadata_list = results.get('metadatas', [])
            ids_list = results.get('ids', [])
            
            # Group by source to avoid duplicates
            sources = {}
            for doc_id, metadata in zip(ids_list, metadata_list):
                if metadata:
                    source = metadata.get('source', 'Unknown')
                    if source not in sources:
                        sources[source] = {
                            'source': source,
                            'title': metadata.get('title', ''),
                            'type': metadata.get('type', 'unknown'),
                            'domain': metadata.get('domain', ''),
                            'chunk_count': 0
                        }
                    sources[source]['chunk_count'] += 1
            
            return list(sources.values())
            
        except Exception as e:
            print(f"Error listing documents: {str(e)}")
            return []
    
    def clear_conversation(self) -> None:
        """Clear the conversation memory."""
        if self.conversation_memory:
            self.conversation_memory.clear()
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        if self.conversation_memory:
            return self.conversation_memory.get_stats()
        return {"conversation_memory": "disabled"}
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation messages."""
        if self.conversation_memory:
            messages = self.conversation_memory.get_recent_messages(count)
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "tokens": msg.tokens
                }
                for msg in messages
            ]
        return []
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get document statistics from the vector store."""
        try:
            # Get collection info
            count = self.collection.count()
            
            return {
                "total_documents": count,
                "vector_db_path": self.config.vector_db_path,
                "documents_path": self.config.documents_path,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap
            }
        except Exception as e:
            return {"error": f"Failed to get document stats: {str(e)}"}
