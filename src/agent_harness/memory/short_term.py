"""Short-term memory: conversation buffer with sliding window and token limits."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from agent_harness.core.message import Message, Role
from agent_harness.memory.base import BaseMemory, MemoryItem
from agent_harness.memory.retrieval import HybridRetriever
from agent_harness.utils.token_counter import count_messages_tokens


class ShortTermMemory(BaseMemory):
    """Conversation buffer that maintains recent message history.

    Strategies:
        - sliding_window: Keep the last N messages
        - token_limit: Keep messages that fit within a token budget

    The system message (if present) is always preserved regardless of strategy.
    """

    def __init__(
        self,
        max_messages: int = 50,
        max_tokens: int = 8000,
        strategy: str = "sliding_window",
        model: str = "gpt-4o",
    ) -> None:
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.model = model
        self._messages: list[Message] = []

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add text content as a user message."""
        await self.add_message(Message.user(content, metadata=metadata or {}))

    async def add_message(self, message: Message) -> None:
        """Add a message to the conversation buffer."""
        self._messages.append(message)
        self._trim()

    async def query(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        """Query messages using hybrid retrieval (TF-IDF + keyword fallback)."""
        items = [
            MemoryItem(
                content=msg.content or "",
                metadata={"role": msg.role.value},
                importance_score=msg.metadata.get("importance_score", 0.5),
                timestamp=msg.created_at,
            )
            for msg in self._messages
            if msg.content
        ]
        if not items:
            return []
        retriever = HybridRetriever()
        return retriever.retrieve(query, items, top_k=top_k)

    async def get_context_messages(self) -> list[Message]:
        """Get messages suitable for LLM context, respecting limits."""
        return list(self._get_trimmed_messages())

    async def clear(self) -> None:
        self._messages.clear()

    async def forget(self, threshold: float = 0.3) -> int:
        """Remove old messages with low weighted score, then apply trim.

        System messages are always preserved.
        """
        now = datetime.now()
        decay_rate = 0.01
        original_count = len(self._messages)

        kept: list[Message] = []
        for msg in self._messages:
            if msg.role == Role.SYSTEM:
                kept.append(msg)
                continue
            hours = (now - msg.created_at).total_seconds() / 3600
            time_decay = math.exp(-decay_rate * hours)
            importance = msg.metadata.get("importance_score", 0.5)
            weighted = time_decay * (0.8 + importance * 0.4)
            if weighted >= threshold:
                kept.append(msg)

        self._messages = kept
        self._trim()
        return original_count - len(self._messages)

    async def size(self) -> int:
        return len(self._messages)

    def _trim(self) -> None:
        """Apply the configured trimming strategy."""
        if self.strategy == "sliding_window":
            self._trim_sliding_window()
        elif self.strategy == "token_limit":
            self._trim_token_limit()

    def _trim_sliding_window(self) -> None:
        """Keep system message + last N messages."""
        if len(self._messages) <= self.max_messages:
            return

        system_msgs = [m for m in self._messages if m.role == Role.SYSTEM]
        non_system = [m for m in self._messages if m.role != Role.SYSTEM]

        # Keep system messages + most recent non-system messages
        keep_count = self.max_messages - len(system_msgs)
        self._messages = system_msgs + non_system[-keep_count:]

    def _trim_token_limit(self) -> None:
        """Keep system message + most recent messages within token budget."""
        system_msgs = [m for m in self._messages if m.role == Role.SYSTEM]
        non_system = [m for m in self._messages if m.role != Role.SYSTEM]

        # Start from most recent, add messages until budget is exceeded
        result: list[Message] = []
        budget = self.max_tokens - count_messages_tokens(system_msgs, self.model)

        for msg in reversed(non_system):
            msg_tokens = count_messages_tokens([msg], self.model)
            if budget - msg_tokens < 0:
                break
            result.append(msg)
            budget -= msg_tokens

        result.reverse()
        self._messages = system_msgs + result

    def _get_trimmed_messages(self) -> list[Message]:
        """Return trimmed message list."""
        return list(self._messages)
