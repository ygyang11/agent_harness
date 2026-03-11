"""Long-term memory backed by vector store for semantic retrieval."""
from __future__ import annotations

import uuid
from typing import Any, Callable, Awaitable

from agent_harness.core.message import Message
from agent_harness.memory.base import BaseMemory, MemoryItem
from agent_harness.memory.storage.base import BaseVectorStore, VectorDocument


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
        doc = VectorDocument(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
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
                score=r.score,
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

    async def size(self) -> int:
        return await self._store.count()
