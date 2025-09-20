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
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

from .config import Config
from .conversation_memory import ConversationMemory
from .web_scraper import WebScraper


class RAGEngine:
    """RAG Engine for document-based question answering."""
    
    def __init__(self, model: str = "llama3-8b-8192", temperature: float = 0.7, 
                 max_tokens: int = 1000, config: Optional[Config] = None):
        """Initialize RAG Engine.
        
        Args:
            model: Groq model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens for response
            config: Configuration object
        """
        self.config = config or Config()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize components
        self._setup_embeddings()
        self._setup_llm()
        self._setup_vectorstore()
        self._setup_qa_chain()
        
        # Initialize conversation memory
        self.conversation_memory = ConversationMemory(
            max_tokens=self.config.max_conversation_tokens,
            summarization_threshold=self.config.summarization_threshold
        ) if self.config.enable_conversation_memory else None
        
        # Initialize web scraper
        self.web_scraper = WebScraper()
    
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
            max_tokens=self.max_tokens,
            groq_api_key=self.config.groq_api_key
        )
    
    def _setup_vectorstore(self):
        """Setup vector store."""
        # Create vector database directory
        os.makedirs(self.config.vector_db_path, exist_ok=True)
        
        # Initialize ChromaDB
        self.vectorstore = Chroma(
            persist_directory=self.config.vector_db_path,
            embedding_function=self.embeddings
        )
        
        # Load documents if they exist
        self._load_documents()
    
    def _load_documents(self):
        """Load documents from the documents directory."""
        documents_path = Path(self.config.documents_path)
        
        if not documents_path.exists():
            print(f"Documents directory {documents_path} not found. Creating it...")
            documents_path.mkdir(parents=True, exist_ok=True)
            return
        
        # Load all supported document types
        documents = []
        for file_path in documents_path.rglob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                    documents.extend(docs)
                elif file_path.suffix.lower() in ['.txt', '.md']:
                    loader = TextLoader(str(file_path))
                    docs = loader.load()
                    documents.extend(docs)
        
        if documents:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            split_docs = text_splitter.split_documents(documents)
            
            # Add to vector store
            self.vectorstore.add_documents(split_docs)
            print(f"Loaded {len(split_docs)} document chunks into vector store.")
        else:
            print("No documents found in the documents directory.")
    
    def _setup_qa_chain(self):
        """Setup question-answering chain."""
        # Create custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": PROMPT}
        )
    
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
            
            # Create enhanced prompt with conversation context
            if conversation_context:
                enhanced_question = f"""Based on our previous conversation:
{conversation_context}

Current question: {question}

Please answer the current question while considering the conversation context."""
            else:
                enhanced_question = question
            
            # Get response from RAG system
            result = self.qa_chain.invoke(enhanced_question)
            
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
            
            # Add to vector store
            self.vectorstore.add_documents(split_docs)
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
            
            # Add to vector store
            self.vectorstore.add_documents(split_docs)
            print(f"Successfully added {len(split_docs)} chunks from {url}")
            return True
            
        except Exception as e:
            print(f"Error adding document from URL '{url}': {str(e)}")
            return False
    
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
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "vector_db_path": self.config.vector_db_path,
                "documents_path": self.config.documents_path,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap
            }
        except Exception as e:
            return {"error": f"Failed to get document stats: {str(e)}"}
