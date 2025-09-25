"""
Tests for logging configuration and functionality.
"""

import pytest
import logging
import tempfile
import os
from unittest.mock import patch, Mock

from src.rag_sample.logging_config import setup_logging, get_logger


class TestLoggingConfig:
    """Test logging configuration functionality."""
    
    def test_setup_logging_default(self) -> None:
        """Test default logging setup."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_logging()
            mock_basic_config.assert_called_once()
    
    def test_setup_logging_with_file(self) -> None:
        """Test logging setup with log file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            log_file = f.name
        
        try:
            with patch('logging.basicConfig') as mock_basic_config, \
                 patch('pathlib.Path.mkdir') as mock_mkdir:
                setup_logging(log_file=log_file)
                mock_mkdir.assert_called_once()
                mock_basic_config.assert_called_once()
        finally:
            os.unlink(log_file)
    
    def test_get_logger(self) -> None:
        """Test getting logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_logger_hierarchy(self) -> None:
        """Test logger hierarchy."""
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")
        
        assert child_logger.parent == parent_logger
        assert child_logger.name == "parent.child"
