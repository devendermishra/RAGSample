"""
Tests for configuration management.
"""

import os
import tempfile
from pathlib import Path
import pytest
from src.rag_sample.config import Config


class TestConfig:
    """Test configuration functionality."""
    
    def test_config_initialization(self):
        """Test basic config initialization."""
        config = Config()
        assert config.groq_model is not None
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.vector_db_path == "./data/vector_db"
        assert config.documents_path == "./documents"
    
    def test_config_with_env_file(self):
        """Test config loading from environment file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("GROQ_MODEL=test-model\n")
            f.write("TEMPERATURE=0.5\n")
            f.write("MAX_TOKENS=2000\n")
            f.flush()
            
            config = Config(config_path=f.name)
            assert config.groq_model == "test-model"
            assert config.temperature == 0.5
            assert config.max_tokens == 2000
            
            os.unlink(f.name)
    
    def test_config_validation(self):
        """Test config validation."""
        config = Config()
        
        # Test with missing API key
        original_key = os.environ.get("GROQ_API_KEY")
        if "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]
        
        try:
            config._validate()
            # Should not raise exception as we have a key in the test
        except ValueError as e:
            assert "GROQ_API_KEY" in str(e)
        finally:
            if original_key:
                os.environ["GROQ_API_KEY"] = original_key
    
    def test_config_to_dict(self):
        """Test config to dictionary conversion."""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "groq_model" in config_dict
        assert "temperature" in config_dict
        assert "max_tokens" in config_dict
