"""Long-term memory with TF-IDF-based semantic retrieval."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from agent_harness.core.message import Message
from agent_harness.memory.base import BaseMemory, MemoryItem
from agent_harness.memory.retrieval import HybridRetriever


class LongTermMemory(BaseMemory):
    """Long-term memory with TF-IDF-based semantic retrieval.

    Stores messages and retrieves them using the shared HybridRetriever,
    consistent with ShortTermMemory and WorkingMemory.
    """

    def __init__(self, max_documents: int = 10000) -> None:
        self._messages: list[Message] = []
        self._max_documents = max_documents

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        meta = dict(metadata) if metadata else {}
        if "importance_score" not in meta:
            meta["importance_score"] = 0.5
        msg = Message.user(content, metadata=meta)
        self._messages.append(msg)
        if len(self._messages) > self._max_documents:
            self._messages = self._messages[-self._max_documents:]

    async def add_message(self, message: Message) -> None:
        if message.content:
            self._messages.append(message)

    async def query(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        if not self._messages:
            return []
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
        return []

    async def clear(self) -> None:
        self._messages.clear()

    async def forget(self, threshold: float = 0.3) -> int:
        now = datetime.now()
        original = len(self._messages)
        kept: list[Message] = []
        for msg in self._messages:
            hours = (now - msg.created_at).total_seconds() / 3600
            time_decay = math.exp(-0.01 * hours)
            importance = msg.metadata.get("importance_score", 0.5)
            weighted = time_decay * (0.8 + importance * 0.4)
            if weighted >= threshold:
                kept.append(msg)
        self._messages = kept
        return original - len(kept)

    async def size(self) -> int:
        return len(self._messages)
