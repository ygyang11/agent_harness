"""Tests for TF-IDF-based LongTermMemory."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from agent_harness.memory.long_term import LongTermMemory
from agent_harness.core.message import Message


class TestLongTermMemory:
    @pytest.mark.asyncio
    async def test_add_and_query(self) -> None:
        mem = LongTermMemory()
        await mem.add("Python is a programming language")
        await mem.add("JavaScript runs in the browser")
        await mem.add("Python supports async programming")
        results = await mem.query("Python programming", top_k=2)
        assert len(results) == 2
        assert "Python" in results[0].content

    @pytest.mark.asyncio
    async def test_forget_removes_old_documents(self) -> None:
        mem = LongTermMemory()
        old_time = datetime.now() - timedelta(hours=500)
        old_msg = Message.user("old data", metadata={"importance_score": 0.1}, created_at=old_time)
        mem._messages.append(old_msg)
        await mem.add("recent data", metadata={"importance_score": 0.9})
        removed = await mem.forget(threshold=0.3)
        assert removed == 1
        assert await mem.size() == 1

    @pytest.mark.asyncio
    async def test_forget_keeps_important_recent(self) -> None:
        mem = LongTermMemory()
        await mem.add("important data", metadata={"importance_score": 0.9})
        removed = await mem.forget(threshold=0.3)
        assert removed == 0

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        mem = LongTermMemory()
        await mem.add("data")
        await mem.clear()
        assert await mem.size() == 0

    @pytest.mark.asyncio
    async def test_size(self) -> None:
        mem = LongTermMemory()
        assert await mem.size() == 0
        await mem.add("data")
        assert await mem.size() == 1

    @pytest.mark.asyncio
    async def test_get_context_messages_empty(self) -> None:
        mem = LongTermMemory()
        msgs = await mem.get_context_messages()
        assert msgs == []

    @pytest.mark.asyncio
    async def test_add_message(self) -> None:
        mem = LongTermMemory()
        msg = Message.user("hello")
        await mem.add_message(msg)
        assert await mem.size() == 1

    @pytest.mark.asyncio
    async def test_query_returns_importance_and_timestamp(self) -> None:
        mem = LongTermMemory()
        await mem.add("Python is great", metadata={"importance_score": 0.9})
        await mem.add("Java is verbose", metadata={"importance_score": 0.3})
        await mem.add("Python async rocks")
        results = await mem.query("Python", top_k=3)
        assert len(results) == 3
        # Results should have importance_score set by retriever
        assert all(r.importance_score is not None for r in results)
