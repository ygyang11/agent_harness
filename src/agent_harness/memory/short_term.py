"""Short-term memory: conversation buffer with optional compression and token trim."""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agent_harness.core.message import Message, Role
from agent_harness.memory.base import BaseMemory, MemoryItem
from agent_harness.memory.retrieval import HybridRetriever
from agent_harness.utils.token_counter import count_messages_tokens

if TYPE_CHECKING:
    from agent_harness.memory.compressor import ContextCompressor

logger = logging.getLogger(__name__)


class ShortTermMemory(BaseMemory):
    """Conversation buffer with optional LLM compression and token-based trim fallback.

    When a ContextCompressor is attached, compression runs first
    (preserving semantics). Token trim only runs as a safety net
    after compression (or if compression fails/is not attached).

    The system message (if present) is always preserved by the trim fallback.
    """

    def __init__(
        self,
        max_tokens: int = 100000,
        model: str = "gpt-4o",
        compressor: ContextCompressor | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.model = model
        self._messages: list[Message] = []
        self.compressor = compressor

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add text content as a user message."""
        await self.add_message(Message.user(content, metadata=metadata or {}))

    async def add_message(self, message: Message) -> None:
        """Append a message. No trimming — trimming happens at the exit point."""
        self._messages.append(message)

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
        results: list[MemoryItem] = retriever.retrieve(query, items, top_k=top_k)
        return results

    async def get_context_messages(self) -> list[Message]:
        """Get messages suitable for LLM context, respecting limits.

        Flow: compress (if attached and threshold hit) → trim fallback.
        """
        if self.compressor and self.compressor.should_compress(
            self._messages, self.max_tokens
        ):
            try:
                self._messages = await self.compressor.compress(self._messages)
            except Exception as e:
                logger.warning(
                    "Context compression failed; falling back to token trim: %s",
                    e,
                    exc_info=True,
                )

        self._trim_by_tokens()
        return list(self._messages)

    async def clear(self) -> None:
        self._messages.clear()

    async def forget(self, threshold: float = 0.3) -> int:
        """Remove old messages with low weighted score.

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
        return original_count - len(self._messages)

    async def size(self) -> int:
        return len(self._messages)

    def _trim_by_tokens(self) -> None:
        """Keep system messages + most recent atomic groups within token budget.

        Uses atomic grouping to avoid splitting tool_call + tool_result pairs.
        Only trims when total tokens exceed max_tokens.
        """
        from agent_harness.memory.compressor import ContextCompressor

        current = count_messages_tokens(self._messages, self.model)
        if current <= self.max_tokens:
            return

        groups = ContextCompressor._group_atomic_pairs(self._messages)
        system_groups = [g for g in groups if g.is_system]
        non_system_groups = [g for g in groups if not g.is_system]

        system_msgs = [m for g in system_groups for m in g.messages]
        budget = self.max_tokens - count_messages_tokens(system_msgs, self.model)
        kept_groups: list[list[Message]] = []

        for group in reversed(non_system_groups):
            group_tokens = count_messages_tokens(group.messages, self.model)
            if budget - group_tokens < 0:
                break
            kept_groups.append(group.messages)
            budget -= group_tokens

        kept_groups.reverse()
        kept_msgs = [m for group in kept_groups for m in group]
        self._messages = system_msgs + kept_msgs
