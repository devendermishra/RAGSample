"""
Configuration management for RAG Sample application.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from .logging_config import get_logger
from .exceptions import ConfigurationError

logger = get_logger(__name__)


class Config:
    """Configuration class for RAG Sample application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default locations.
        """
        # Load environment variables
        if config_path:
            load_dotenv(config_path)
        else:
            # Try to load from default locations
            load_dotenv()  # .env in current directory
            load_dotenv(Path.home() / ".rag_sample" / ".env")  # User config
        
        # LLM configuration - support multiple providers
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm_model = os.getenv("LLM_MODEL")  # New unified model variable
        self.groq_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")  # Keep for backward compatibility
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
        self.vector_db_collection = os.getenv("VECTOR_DB_COLLECTION", "rag_sample_collection")
        
        # Vector database settings
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/_vector_db_old")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Document processing settings
        self.documents_path = os.getenv("DOCUMENTS_PATH", "./documents")
        
        # Conversation memory settings
        self.max_conversation_tokens = int(os.getenv("MAX_CONVERSATION_TOKENS", "1000"))  # Reduced from 4000 to 1000
        self.summarization_threshold = float(os.getenv("SUMMARIZATION_THRESHOLD", "0.6"))  # Reduced from 0.8 to 0.6
        self.enable_conversation_memory = os.getenv("ENABLE_CONVERSATION_MEMORY", "true").lower() == "true"
        
        # Document retrieval settings
        self.retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
        self.retrieval_threshold = float(os.getenv("RETRIEVAL_THRESHOLD", "0.3"))
        self.enable_retrieval_debug = os.getenv("ENABLE_RETRIEVAL_DEBUG", "false").lower() == "true"
        
        # UI settings
        self.user_prompt = os.getenv("USER_PROMPT", "You")
        self.goodbye_message = os.getenv("GOODBYE_MESSAGE", "Goodbye! Thanks for using RAG Sample.")
        self.welcome_message = os.getenv("WELCOME_MESSAGE", "Welcome to RAG Sample! Ask me anything about your documents.")
        
        # Validate required settings
        self._validate()
    
    def _validate(self):
        """Validate configuration settings."""
        logger.info("Validating configuration settings")
        
        # Check for at least one API key
        if not any([self.openai_api_key, self.google_api_key, self.groq_api_key]):
            error_msg = (
                "At least one API key is required. Please set one of the following:\n"
                "  - OPENAI_API_KEY (for OpenAI models)\n"
                "  - GOOGLE_API_KEY (for Gemini models)\n"
                "  - GROQ_API_KEY (for Groq models)\n"
                "Priority order: OPENAI_API_KEY > GOOGLE_API_KEY > GROQ_API_KEY"
            )
            logger.error(error_msg)
            raise ConfigurationError(error_msg, "MISSING_API_KEY")
        
        logger.info("Configuration validation successful")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "groq_model": self.groq_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "vector_db_path": self.vector_db_path,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "documents_path": self.documents_path,
            "max_conversation_tokens": self.max_conversation_tokens,
            "summarization_threshold": self.summarization_threshold,
            "enable_conversation_memory": self.enable_conversation_memory,
            "retrieval_top_k": self.retrieval_top_k,
            "retrieval_threshold": self.retrieval_threshold,
            "enable_retrieval_debug": self.enable_retrieval_debug,
            "user_prompt": self.user_prompt,
            "goodbye_message": self.goodbye_message,
            "welcome_message": self.welcome_message,
        }
