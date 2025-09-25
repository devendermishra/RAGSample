"""
Tests for CLI functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from src.rag_sample.cli import main, CommandHistory
from src.rag_sample.config import Config
from src.rag_sample.rag_engine import RAGEngine
from src.rag_sample.exceptions import RAGSampleError


class TestCLI:
    """Test CLI functionality."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.mock_config = Mock(spec=Config)
        self.mock_rag_engine = Mock(spec=RAGEngine)
    
    def test_main_command_help(self) -> None:
        """Test main command help."""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert "RAG Sample CLI" in result.output
    
    def test_main_command_with_model(self) -> None:
        """Test main command with model option."""
        with patch('src.rag_sample.cli.Config', return_value=self.mock_config), \
             patch('src.rag_sample.cli.RAGEngine', return_value=self.mock_rag_engine):
            
            result = self.runner.invoke(main, ['--model', 'llama-3.1-8b-instant'])
            assert result.exit_code == 0
    
    def test_main_command_with_temperature(self) -> None:
        """Test main command with temperature option."""
        with patch('src.rag_sample.cli.Config', return_value=self.mock_config), \
             patch('src.rag_sample.cli.RAGEngine', return_value=self.mock_rag_engine):
            
            result = self.runner.invoke(main, ['--temperature', '0.5'])
            assert result.exit_code == 0
    
    def test_main_command_with_max_tokens(self) -> None:
        """Test main command with max tokens option."""
        with patch('src.rag_sample.cli.Config', return_value=self.mock_config), \
             patch('src.rag_sample.cli.RAGEngine', return_value=self.mock_rag_engine):
            
            result = self.runner.invoke(main, ['--max-tokens', '2000'])
            assert result.exit_code == 0
    
    def test_main_command_with_debug(self) -> None:
        """Test main command with debug option."""
        with patch('src.rag_sample.cli.Config', return_value=self.mock_config), \
             patch('src.rag_sample.cli.RAGEngine', return_value=self.mock_rag_engine):
            
            result = self.runner.invoke(main, ['--debug'])
            assert result.exit_code == 0
    
    def test_main_command_with_help(self) -> None:
        """Test main command with help option."""
        with patch('src.rag_sample.cli.Config', return_value=self.mock_config), \
             patch('src.rag_sample.cli.RAGEngine', return_value=self.mock_rag_engine):
            
            result = self.runner.invoke(main, ['--help'])
            assert result.exit_code == 0
    
    def test_main_command_with_version(self) -> None:
        """Test main command with version option."""
        with patch('src.rag_sample.cli.Config', return_value=self.mock_config), \
             patch('src.rag_sample.cli.RAGEngine', return_value=self.mock_rag_engine):
            
            result = self.runner.invoke(main, ['--version'])
            assert result.exit_code == 0


class TestCommandHistory:
    """Test command history functionality."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.history = CommandHistory()
    
    def test_command_history_initialization(self) -> None:
        """Test command history initialization."""
        assert self.history.history_file is not None
        assert isinstance(self.history.history_file, str)
    
    def test_add_command(self) -> None:
        """Test adding command to history."""
        self.history.add_command("test command")
        assert "test command" in self.history.history
    
    def test_get_history(self) -> None:
        """Test getting command history."""
        self.history.add_command("command1")
        self.history.add_command("command2")
        
        history = self.history.get_history()
        assert len(history) == 2
        assert "command1" in history
        assert "command2" in history
    
    def test_clear_history(self) -> None:
        """Test clearing command history."""
        self.history.add_command("test command")
        assert len(self.history.history) == 1
        
        self.history.clear_history()
        assert len(self.history.history) == 0
    
    def test_save_history(self) -> None:
        """Test saving history to file."""
        with patch('builtins.open', Mock()) as mock_open:
            self.history.add_command("test command")
            self.history.save_history()
            mock_open.assert_called_once()
    
    def test_load_history(self) -> None:
        """Test loading history from file."""
        with patch('builtins.open', Mock()) as mock_open, \
             patch('os.path.exists', return_value=True):
            
            mock_file = Mock()
            mock_file.__enter__.return_value = mock_file
            mock_file.__exit__.return_value = None
            mock_file.readlines.return_value = ["command1\n", "command2\n"]
            mock_open.return_value = mock_file
            
            self.history.load_history()
            assert len(self.history.history) == 2
            assert "command1" in self.history.history
            assert "command2" in self.history.history
