"""Long-term memory backed by vector store for semantic retrieval."""
from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime
from typing import Any, Callable, Awaitable

from agent_harness.core.message import Message
from agent_harness.memory.base import BaseMemory, MemoryItem
from agent_harness.memory.storage.base import BaseVectorStore, VectorDocument

logger = logging.getLogger(__name__)

# Type alias for embedding function
EmbeddingFn = Callable[[str], Awaitable[list[float]]]


class LongTermMemory(BaseMemory):
    """Semantic retrieval memory backed by a vector store.

    Stores content with embeddings for similarity-based retrieval.
    Ideal for research scenarios where large amounts of information
    need to be stored and retrieved by relevance.

    Requires an embedding function to be injected (e.g., OpenAI embeddings).

    Example:
        async def embed(text: str) -> list[float]:
            response = await openai.embeddings.create(input=text, model="text-embedding-3-small")
            return response.data[0].embedding

        memory = LongTermMemory(store=NumpyVectorStore(), embedding_fn=embed)
        await memory.add("AI safety is important", metadata={"source": "paper"})
        results = await memory.query("safety research", top_k=3)
    """

    def __init__(
        self,
        store: BaseVectorStore,
        embedding_fn: EmbeddingFn,
    ) -> None:
        self._store = store
        self._embed = embedding_fn

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add content to long-term memory with auto-generated embedding."""
        doc_id = uuid.uuid4().hex
        embedding = await self._embed(content)
        meta = metadata or {}
        if "created_at" not in meta:
            meta["created_at"] = datetime.now().isoformat()
        doc = VectorDocument(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=meta,
        )
        await self._store.upsert([doc])

    async def add_message(self, message: Message) -> None:
        """Add a message's content to long-term memory."""
        if message.content:
            await self.add(
                message.content,
                metadata={"role": message.role.value},
            )

    async def query(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        """Query long-term memory by semantic similarity."""
        query_embedding = await self._embed(query)
        results = await self._store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
        return [
            MemoryItem(
                content=r.document.content,
                metadata=r.document.metadata,
                importance_score=r.score,
            )
            for r in results
        ]

    async def get_context_messages(self) -> list[Message]:
        """Long-term memory doesn't provide context messages directly.

        Use query() to retrieve relevant content and inject manually.
        Returns empty list.
        """
        return []

    async def clear(self) -> None:
        await self._store.clear()

    async def forget(self, threshold: float = 0.3) -> int:
        """Remove long-term memories with low weighted score.

        Uses ``list_all()`` on the vector store to scan documents.
        If the store does not support iteration, returns 0.
        """
        docs = await self._store.list_all()
        if not docs:
            logger.debug("LongTermMemory.forget: no docs to evaluate (threshold=%.2f)", threshold)
            return 0

        now = datetime.now()
        decay_rate = 0.01
        ids_to_remove: list[str] = []

        for doc in docs:
            created_str = doc.metadata.get("created_at")
            if created_str:
                try:
                    created = datetime.fromisoformat(created_str)
                except (ValueError, TypeError):
                    created = now
            else:
                created = now

            hours = (now - created).total_seconds() / 3600
            time_decay = math.exp(-decay_rate * hours)
            importance = doc.metadata.get("importance_score", 0.5)
            weighted = time_decay * (0.8 + importance * 0.4)
            if weighted < threshold:
                ids_to_remove.append(doc.id)

        if ids_to_remove:
            await self._store.delete(ids_to_remove)
            logger.info("LongTermMemory.forget: removed %d documents", len(ids_to_remove))

        return len(ids_to_remove)

    async def size(self) -> int:
        return await self._store.count()
