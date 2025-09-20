"""
Conversation memory system for maintaining chat context and handling token limits.
"""

import tiktoken
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    tokens: int = 0
    
    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = self._count_tokens(self.content)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough estimation (4 chars per token)
            return len(text) // 4


class ConversationMemory:
    """Manages conversation history with token tracking and summarization."""
    
    def __init__(self, max_tokens: int = 4000, summarization_threshold: float = 0.8):
        """Initialize conversation memory.
        
        Args:
            max_tokens: Maximum tokens to maintain in conversation
            summarization_threshold: When to trigger summarization (0.0-1.0)
        """
        self.max_tokens = max_tokens
        self.summarization_threshold = summarization_threshold
        self.messages: List[Message] = []
        self.summary: str = ""
        self.current_tokens = 0
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now()
        )
        
        self.messages.append(message)
        self.current_tokens += message.tokens
        
        # Check if we need to summarize
        if self._should_summarize():
            self._summarize_conversation()
    
    def _should_summarize(self) -> bool:
        """Check if conversation should be summarized."""
        threshold_tokens = int(self.max_tokens * self.summarization_threshold)
        return self.current_tokens > threshold_tokens
    
    def _summarize_conversation(self) -> None:
        """Summarize the conversation to reduce token count."""
        if len(self.messages) < 2:
            return
        
        # Create summary of older messages
        older_messages = self.messages[:-2]  # Keep last 2 messages
        if older_messages:
            # Create a simple summary
            user_messages = [msg.content for msg in older_messages if msg.role == 'user']
            assistant_messages = [msg.content for msg in older_messages if msg.role == 'assistant']
            
            summary_parts = []
            if user_messages:
                summary_parts.append(f"User asked about: {'; '.join(user_messages[:3])}")
            if assistant_messages:
                summary_parts.append(f"Assistant provided information about: {'; '.join(assistant_messages[:3])}")
            
            self.summary = "Previous conversation: " + ". ".join(summary_parts)
            
            # Remove older messages and update token count
            removed_tokens = sum(msg.tokens for msg in older_messages)
            self.messages = self.messages[-2:]  # Keep only last 2 messages
            self.current_tokens -= removed_tokens
            
            # Add summary tokens
            summary_tokens = self._count_tokens(self.summary)
            self.current_tokens += summary_tokens
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return len(text) // 4
    
    def get_conversation_context(self) -> str:
        """Get the current conversation context for the LLM."""
        context_parts = []
        
        # Add summary if available
        if self.summary:
            context_parts.append(self.summary)
        
        # Add recent messages
        for message in self.messages:
            role_label = "User" if message.role == "user" else "Assistant"
            context_parts.append(f"{role_label}: {message.content}")
        
        return "\n".join(context_parts)
    
    def get_recent_messages(self, count: int = 5) -> List[Message]:
        """Get the most recent messages."""
        return self.messages[-count:] if self.messages else []
    
    def clear(self) -> None:
        """Clear the conversation memory."""
        self.messages.clear()
        self.summary = ""
        self.current_tokens = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return {
            "total_messages": len(self.messages),
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "has_summary": bool(self.summary),
            "token_usage_percentage": (self.current_tokens / self.max_tokens) * 100
        }
