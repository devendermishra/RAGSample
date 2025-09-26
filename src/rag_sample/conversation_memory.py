"""
Conversation memory system for maintaining chat context and handling token limits.
"""

import tiktoken
from typing import List, Dict, Any
from dataclasses import dataclass

from .logging_config import get_logger

logger = get_logger(__name__)
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


def _create_summarization_prompt(conversation_text: str) -> str:
    """Create advanced summarization prompt using Ready Tensor techniques."""
    return f"""You are a conversation summarizer. Your task is to create a concise summary that preserves:

1. Key topics discussed
2. Important decisions made
3. Critical information shared
4. User preferences or requirements
5. Context needed for future interactions

Conversation to summarize:
{conversation_text}

Create a summary that:
- Preserves essential context for future responses
- Maintains user intent and preferences
- Captures key technical details
- Is concise but comprehensive

Summary:"""


def _format_conversation_for_summary(messages: List[Message]) -> str:
    """Format conversation for summarization using Ready Tensor techniques."""
    formatted_messages = []
    for msg in messages:
        timestamp = msg.timestamp.strftime("%H:%M:%S")
        formatted_messages.append(f"[{timestamp}] {msg.role.upper()}: {msg.content}")
    return "\n".join(formatted_messages)


def _count_tokens(text: str) -> int:
    """Count tokens in text."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


class ConversationMemory:
    """Manages conversation history with token tracking and summarization."""
    
    def __init__(self, max_tokens: int = 4000, summarization_threshold: float = 0.8, llm=None):
        """Initialize conversation memory.
        
        Args:
            max_tokens: Maximum tokens to maintain in conversation
            summarization_threshold: When to trigger summarization (0.0-1.0)
            llm: Language model for summarization
        """
        self.max_tokens = max_tokens
        self.summarization_threshold = summarization_threshold
        self.llm = llm
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
        return self.current_tokens >= threshold_tokens
    
    def _summarize_conversation(self) -> None:
        """Summarize the conversation using Ready Tensor advanced techniques."""
        if len(self.messages) < 2:
            return
        
        # Create summary of older messages
        older_messages = self.messages[:-2]  # Keep last 2 messages
        if older_messages:
            if self.llm:
                # Use Ready Tensor's advanced summarization approach
                conversation_text = _format_conversation_for_summary(older_messages)
                summary_prompt = _create_summarization_prompt(conversation_text)
                
                try:
                    # Use the same system prompt as the main RAG system
                    response = self.llm.invoke(summary_prompt)
                    if hasattr(response, 'content'):
                        self.summary = f"Previous conversation: {response.content}"
                    else:
                        self.summary = f"Previous conversation: {str(response)}"
                except Exception as e:
                    # Fallback to simple summary if LLM fails
                    self._create_simple_summary(older_messages)
            else:
                # Fallback to simple summary if no LLM provided
                self._create_simple_summary(older_messages)
            
            # Remove older messages and update token count
            removed_tokens = sum(msg.tokens for msg in older_messages)
            self.messages = self.messages[-2:]  # Keep only last 2 messages
            self.current_tokens -= removed_tokens
            
            # Add summary tokens
            summary_tokens = _count_tokens(self.summary)
            self.current_tokens += summary_tokens

    def _create_simple_summary(self, older_messages: List[Message]) -> None:
        """Create a simple summary as fallback."""
        user_messages = [msg.content for msg in older_messages if msg.role == 'user']
        assistant_messages = [msg.content for msg in older_messages if msg.role == 'assistant']
        
        summary_parts = []
        if user_messages:
            summary_parts.append(f"User asked about: {'; '.join(user_messages[:3])}")
        if assistant_messages:
            summary_parts.append(f"Assistant provided information about: {'; '.join(assistant_messages[:3])}")
        
        self.summary = "Previous conversation: " + ". ".join(summary_parts)

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
