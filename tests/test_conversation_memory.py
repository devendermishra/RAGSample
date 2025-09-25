"""
Tests for conversation memory functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.rag_sample.conversation_memory import ConversationMemory, Message
from src.rag_sample.exceptions import ConversationMemoryError


class TestConversationMemory:
    """Test conversation memory functionality."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.memory = ConversationMemory(
            max_tokens=1000,
            summarization_threshold=0.8,
            llm=self.mock_llm
        )
    
    def test_conversation_memory_initialization(self) -> None:
        """Test conversation memory initialization."""
        assert self.memory.max_tokens == 1000
        assert self.memory.summarization_threshold == 0.8
        assert self.memory.llm == self.mock_llm
        assert len(self.memory.messages) == 0
        assert self.memory.current_tokens == 0
    
    def test_add_message(self) -> None:
        """Test adding message to memory."""
        self.memory.add_message("user", "Hello")
        assert len(self.memory.messages) == 1
        assert self.memory.messages[0].role == "user"
        assert self.memory.messages[0].content == "Hello"
    
    def test_add_message_with_timestamp(self) -> None:
        """Test adding message with timestamp."""
        with patch('src.rag_sample.conversation_memory.datetime') as mock_datetime:
            mock_datetime.now.return_value = Mock(isoformat=lambda: "2024-01-01T00:00:00")
            
            self.memory.add_message("user", "Hello")
            assert self.memory.messages[0].timestamp == "2024-01-01T00:00:00"
    
    def test_get_conversation_context(self) -> None:
        """Test getting conversation context."""
        self.memory.add_message("user", "Hello")
        self.memory.add_message("assistant", "Hi there!")
        
        context = self.memory.get_conversation_context()
        assert "Hello" in context
        assert "Hi there!" in context
    
    def test_get_conversation_context_empty(self) -> None:
        """Test getting conversation context when empty."""
        context = self.memory.get_conversation_context()
        assert context == ""
    
    def test_get_conversation_context_with_summarization(self) -> None:
        """Test getting conversation context with summarization."""
        # Add many messages to trigger summarization
        for i in range(10):
            self.memory.add_message("user", f"Message {i}")
            self.memory.add_message("assistant", f"Response {i}")
        
        # Mock the summarization
        self.mock_llm.invoke.return_value = Mock(content="Summarized conversation")
        
        with patch.object(self.memory, '_summarize_conversation', return_value="Summarized conversation"):
            context = self.memory.get_conversation_context()
            assert "Summarized conversation" in context
    
    def test_get_stats(self) -> None:
        """Test getting conversation statistics."""
        self.memory.add_message("user", "Hello")
        self.memory.add_message("assistant", "Hi there!")
        
        stats = self.memory.get_stats()
        assert stats["total_messages"] == 2
        assert stats["current_tokens"] == self.memory.current_tokens
        assert stats["max_tokens"] == 1000
        assert stats["summarization_threshold"] == 0.8
        assert stats["needs_summarization"] == (self.memory.current_tokens > 1000 * 0.8)
    
    def test_clear_memory(self) -> None:
        """Test clearing conversation memory."""
        self.memory.add_message("user", "Hello")
        self.memory.add_message("assistant", "Hi there!")
        
        assert len(self.memory.messages) == 2
        
        self.memory.clear_memory()
        assert len(self.memory.messages) == 0
        assert self.memory.current_tokens == 0
    
    def test_summarize_conversation_success(self) -> None:
        """Test successful conversation summarization."""
        # Add some messages
        self.memory.add_message("user", "Hello")
        self.memory.add_message("assistant", "Hi there!")
        
        # Mock the LLM response
        self.mock_llm.invoke.return_value = Mock(content="Summarized conversation")
        
        summary = self.memory._summarize_conversation()
        assert summary == "Summarized conversation"
        self.mock_llm.invoke.assert_called_once()
    
    def test_summarize_conversation_llm_error(self) -> None:
        """Test conversation summarization when LLM fails."""
        # Add some messages
        self.memory.add_message("user", "Hello")
        self.memory.add_message("assistant", "Hi there!")
        
        # Mock the LLM to raise an error
        self.mock_llm.invoke.side_effect = Exception("LLM error")
        
        summary = self.memory._summarize_conversation()
        assert "Summarized conversation" in summary  # Should fall back to simple summary
    
    def test_format_conversation_for_summary(self) -> None:
        """Test formatting conversation for summary."""
        self.memory.add_message("user", "Hello")
        self.memory.add_message("assistant", "Hi there!")
        
        formatted = self.memory._format_conversation_for_summary()
        assert "User: Hello" in formatted
        assert "Assistant: Hi there!" in formatted
    
    def test_create_summarization_prompt(self) -> None:
        """Test creating summarization prompt."""
        conversation = "User: Hello\nAssistant: Hi there!"
        
        prompt = self.memory._create_summarization_prompt(conversation)
        assert "Summarize the following conversation" in prompt
        assert "User: Hello" in prompt
        assert "Assistant: Hi there!" in prompt
    
    def test_token_counting(self) -> None:
        """Test token counting functionality."""
        # Mock tiktoken
        with patch('tiktoken.get_encoding') as mock_get_encoding:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_get_encoding.return_value = mock_encoding
            
            self.memory.add_message("user", "Test message")
            assert self.memory.current_tokens == 5
    
    def test_token_counting_error(self) -> None:
        """Test token counting when tiktoken fails."""
        with patch('tiktoken.get_encoding', side_effect=Exception("Token error")):
            # Should not raise an error, just use 0 tokens
            self.memory.add_message("user", "Test message")
            assert self.memory.current_tokens == 0
    
    def test_needs_summarization(self) -> None:
        """Test needs summarization logic."""
        # Set a low threshold for testing
        self.memory.summarization_threshold = 0.1
        
        # Add a message that would trigger summarization
        with patch('tiktoken.get_encoding') as mock_get_encoding:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1] * 200  # 200 tokens
            mock_get_encoding.return_value = mock_encoding
            
            self.memory.add_message("user", "Test message")
            assert self.memory.current_tokens == 200
            assert self.memory.current_tokens > self.memory.max_tokens * self.memory.summarization_threshold
