"""
Configuration management for RAG Sample application.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv


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
        
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
        
        # Vector database settings
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Document processing settings
        self.documents_path = os.getenv("DOCUMENTS_PATH", "./documents")
        
        # Conversation memory settings
        self.max_conversation_tokens = int(os.getenv("MAX_CONVERSATION_TOKENS", "4000"))
        self.summarization_threshold = float(os.getenv("SUMMARIZATION_THRESHOLD", "0.8"))
        self.enable_conversation_memory = os.getenv("ENABLE_CONVERSATION_MEMORY", "true").lower() == "true"
        
        # Validate required settings
        self._validate()
    
    def _validate(self):
        """Validate configuration settings."""
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is required. Please set it in your environment "
                "or create a .env file with GROQ_API_KEY=your_key_here"
            )
    
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
        }
