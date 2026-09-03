"""
RAG Agent with Database Routing - Conversation Memory System.

Provides short-term session conversation history with rolling turn windowing
and context formatting for multi-turn RAG queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ConversationMemory:
    """Manages short-term multi-turn conversation memory for a session."""

    max_turns: int = 5
    messages: list[ChatMessage] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        """Record user message."""
        self.messages.append(ChatMessage(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        """Record assistant message."""
        self.messages.append(ChatMessage(role="assistant", content=content))
        self._trim()

    def _trim(self) -> None:
        """Keep memory within max_turns limit (2 messages per turn)."""
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()

    def format_history_context(self) -> str:
        """Format past conversation turns into a context block for the LLM/router."""
        if not self.messages:
            return ""
        turns = []
        # The current query is appended AFTER generation (see pipeline), so
        # everything stored here is genuine history - include all of it.
        for msg in self.messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            turns.append(f"{role_label}: {msg.content}")
        return "\n".join(turns)
