"""
RAG Engine implementation for document chat functionality.
"""

import os
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from .config import Config
from .conversation_memory import ConversationMemory
from .prompt_builder import PromptManager
from .document_manager import DocumentManager
from .retrieval_engine import RetrievalEngine
from .logging_config import get_logger

logger = get_logger(__name__)


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
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()
        
        # Initialize document and retrieval managers
        self.document_manager = DocumentManager(self.config, self.chroma_client, self.collection)
        self.retrieval_engine = RetrievalEngine(self.config, self.embeddings, self.collection)
        
        # Load documents if not already present
        self.document_manager.load_documents()
    
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
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config.vector_db_collection
        )
    
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
        logger.info(f"Processing question: {question[:100]}...")
        
        try:
            # Add user question to conversation memory
            if self.conversation_memory:
                self.conversation_memory.add_message("user", question)
            
            # Get conversation context if available
            conversation_context = ""
            if self.conversation_memory:
                conversation_context = self.conversation_memory.get_conversation_context()
            
            # STEP 1: Retrieve relevant documents first
            retrieved_docs = self.retrieval_engine.retrieve_documents(question)
            
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
    
    def add_document(self, file_path: str) -> bool:
        """Add a new document to the vector store."""
        return self.document_manager.add_document(file_path)
    
    def add_document_from_url(self, url: str) -> bool:
        """Add a document from a URL."""
        return self.document_manager.add_document_from_url(url)
    
    def remove_document(self, identifier: str) -> bool:
        """Remove a document from the vector store."""
        return self.document_manager.remove_document(identifier)
    
    def list_documents(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all documents in the vector store."""
        return self.document_manager.list_documents()
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about documents in the vector store."""
        return self.document_manager.get_document_stats()
    
    def reload_documents(self) -> bool:
        """Force reload all documents from the documents directory."""
        return self.document_manager.reload_documents()
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics.
        
        Returns:
            Dictionary with conversation statistics
        """
        if not self.conversation_memory:
            return {"error": "Conversation memory is disabled"}
        
        return {
            "total_messages": len(self.conversation_memory.messages),
            "current_tokens": self.conversation_memory.get_current_token_count(),
            "max_tokens": self.conversation_memory.max_tokens,
            "summarization_threshold": self.conversation_memory.summarization_threshold,
            "memory_enabled": self.config.enable_conversation_memory
        }
    
    def clear_conversation(self):
        """Clear the conversation memory."""
        if self.conversation_memory:
            self.conversation_memory.messages = []
            self.conversation_memory.current_tokens = 0
            logger.info("Conversation memory cleared")
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages from conversation memory.
        
        Args:
            count: Number of recent messages to return
            
        Returns:
            List of recent messages
        """
        if not self.conversation_memory:
            return []
        
        recent_messages = self.conversation_memory.messages[-count:]
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat') else str(msg.timestamp)
            }
            for msg in recent_messages
        ]
    
    def test_url(self, url: str) -> str:
        """Test if a URL can be scraped and processed."""
        return self.document_manager.test_url(url)
    
    def get_retrieval_settings(self) -> Dict[str, Any]:
        """Get current retrieval settings.
        
        Returns:
            Dictionary with retrieval settings
        """
        return {
            "top_k": self.config.retrieval_top_k,
            "threshold": self.config.retrieval_threshold,
            "debug": self.config.enable_retrieval_debug
        }
    
    def set_retrieval_settings(self, top_k: int, threshold: float):
        """Set retrieval settings.
        
        Args:
            top_k: Number of documents to retrieve
            threshold: Similarity threshold for filtering
        """
        self.config.retrieval_top_k = top_k
        self.config.retrieval_threshold = threshold
        logger.info(f"Retrieval settings updated: top_k={top_k}, threshold={threshold}")
    
    def toggle_debug(self):
        """Toggle debug mode for retrieval."""
        self.config.enable_retrieval_debug = not self.config.enable_retrieval_debug
        logger.info(f"Debug mode toggled: {self.config.enable_retrieval_debug}")
    
    def set_ui_settings(self, user_prompt: str, goodbye_message: str):
        """Set UI settings.
        
        Args:
            user_prompt: User prompt text
            goodbye_message: Goodbye message text
        """
        self.config.user_prompt = user_prompt
        self.config.goodbye_message = goodbye_message
        logger.info("UI settings updated")
    
    @property
    def debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self.config.enable_retrieval_debug