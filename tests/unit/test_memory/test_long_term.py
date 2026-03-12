"""Regression tests for LongTermMemory.forget() fix."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from agent_harness.memory.long_term import LongTermMemory
from agent_harness.memory.storage.numpy_store import NumpyVectorStore
from agent_harness.memory.storage.base import VectorDocument


class TestLongTermMemoryForget:
    """Issue #5: LongTermMemory.forget() must actually remove documents."""

    @staticmethod
    async def _dummy_embed(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_forget_removes_old_documents(self) -> None:
        """Documents with old created_at should be removed."""
        store = NumpyVectorStore()
        memory = LongTermMemory(store=store, embedding_fn=self._dummy_embed)

        # Insert documents with old timestamps
        old_time = (datetime.now() - timedelta(days=365)).isoformat()
        doc = VectorDocument(
            id="old_doc",
            content="old data",
            embedding=[0.1, 0.2, 0.3],
            metadata={"created_at": old_time, "importance_score": 0.1},
        )
        await store.upsert([doc])

        # Insert a recent document
        await memory.add("recent data")

        assert await store.count() == 2
        removed = await memory.forget(threshold=0.3)
        assert removed >= 1
        assert await store.count() < 2

    @pytest.mark.asyncio
    async def test_forget_keeps_important_recent_documents(self) -> None:
        """Recent documents with high importance should survive forget."""
        store = NumpyVectorStore()
        memory = LongTermMemory(store=store, embedding_fn=self._dummy_embed)

        await memory.add("important recent data")

        removed = await memory.forget(threshold=0.3)
        assert removed == 0
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_list_all_returns_all_documents(self) -> None:
        """NumpyVectorStore.list_all() should return all stored docs."""
        store = NumpyVectorStore()
        docs = [
            VectorDocument(id=f"doc_{i}", content=f"content {i}", embedding=[0.1 * i])
            for i in range(5)
        ]
        await store.upsert(docs)
        all_docs = await store.list_all()
        assert len(all_docs) == 5
