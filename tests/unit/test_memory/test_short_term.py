"""Tests for agent_harness.memory.short_term — ShortTermMemory buffer and trimming."""
from __future__ import annotations

import pytest

from agent_harness.core.message import Message, Role
from agent_harness.memory.short_term import ShortTermMemory


class TestShortTermMemoryBasic:
    @pytest.mark.asyncio
    async def test_add_message(self) -> None:
        mem = ShortTermMemory()
        msg = Message.user("hello")
        await mem.add_message(msg)
        assert await mem.size() == 1

    @pytest.mark.asyncio
    async def test_add_text(self) -> None:
        mem = ShortTermMemory()
        await mem.add("some text")
        assert await mem.size() == 1
        msgs = await mem.get_context_messages()
        assert msgs[0].role == Role.USER
        assert msgs[0].content == "some text"

    @pytest.mark.asyncio
    async def test_get_context_messages(self) -> None:
        mem = ShortTermMemory()
        await mem.add_message(Message.system("You are helpful."))
        await mem.add_message(Message.user("Hi"))
        await mem.add_message(Message.assistant("Hello!"))
        msgs = await mem.get_context_messages()
        assert len(msgs) == 3
        assert msgs[0].role == Role.SYSTEM
        assert msgs[1].role == Role.USER
        assert msgs[2].role == Role.ASSISTANT

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        mem = ShortTermMemory()
        await mem.add_message(Message.user("a"))
        await mem.add_message(Message.user("b"))
        assert await mem.size() == 2
        await mem.clear()
        assert await mem.size() == 0
        assert await mem.get_context_messages() == []

    @pytest.mark.asyncio
    async def test_query_returns_relevant(self) -> None:
        mem = ShortTermMemory()
        for i in range(10):
            await mem.add_message(Message.user(f"msg-{i}"))
        items = await mem.query("msg", top_k=3)
        assert len(items) == 3
        # All returned items should be from the memory
        assert all("msg-" in item.content for item in items)


class TestSlidingWindowTrim:
    @pytest.mark.asyncio
    async def test_trim_keeps_max_messages(self) -> None:
        mem = ShortTermMemory(max_messages=3)
        for i in range(5):
            await mem.add_message(Message.user(f"msg-{i}"))
        msgs = await mem.get_context_messages()
        assert len(msgs) == 3
        assert msgs[0].content == "msg-2"
        assert msgs[-1].content == "msg-4"

    @pytest.mark.asyncio
    async def test_system_message_preserved(self) -> None:
        mem = ShortTermMemory(max_messages=3)
        await mem.add_message(Message.system("System prompt"))
        for i in range(5):
            await mem.add_message(Message.user(f"msg-{i}"))
        msgs = await mem.get_context_messages()
        # System message + 2 most recent non-system
        assert len(msgs) == 3
        assert msgs[0].role == Role.SYSTEM
        assert msgs[0].content == "System prompt"

    @pytest.mark.asyncio
    async def test_no_trim_when_under_limit(self) -> None:
        mem = ShortTermMemory(max_messages=10)
        await mem.add_message(Message.user("one"))
        await mem.add_message(Message.user("two"))
        msgs = await mem.get_context_messages()
        assert len(msgs) == 2

    @pytest.mark.asyncio
    async def test_multiple_system_messages(self) -> None:
        mem = ShortTermMemory(max_messages=4)
        await mem.add_message(Message.system("sys1"))
        await mem.add_message(Message.system("sys2"))
        for i in range(5):
            await mem.add_message(Message.user(f"u{i}"))
        msgs = await mem.get_context_messages()
        assert len(msgs) == 4
        sys_msgs = [m for m in msgs if m.role == Role.SYSTEM]
        assert len(sys_msgs) == 2


class TestSlidingWindowDefaults:
    @pytest.mark.asyncio
    async def test_default_strategy_is_sliding_window(self) -> None:
        mem = ShortTermMemory()
        assert mem.strategy == "sliding_window"

    @pytest.mark.asyncio
    async def test_default_max_messages(self) -> None:
        mem = ShortTermMemory()
        assert mem.max_messages == 50


class TestForgetImportanceScore:
    """Tests for forget() using importance_score from message metadata."""

    @pytest.mark.asyncio
    async def test_high_importance_retained(self) -> None:
        """A message with high importance_score should be retained after forget."""
        mem = ShortTermMemory(max_messages=50)
        msg = Message.user("important", metadata={"importance_score": 0.9})
        await mem.add_message(msg)
        forgotten = await mem.forget(threshold=0.3)
        assert forgotten == 0
        assert await mem.size() == 1

    @pytest.mark.asyncio
    async def test_low_importance_forgotten(self) -> None:
        """A message with low importance_score should be more easily forgotten."""
        mem = ShortTermMemory(max_messages=50)
        msg = Message.user("trivial", metadata={"importance_score": 0.1})
        await mem.add_message(msg)
        # Use a high threshold so the low-importance message gets removed
        forgotten = await mem.forget(threshold=1.1)
        assert forgotten == 1
        assert await mem.size() == 0

    @pytest.mark.asyncio
    async def test_default_importance_when_missing(self) -> None:
        """A message with no importance_score defaults to 0.5."""
        mem = ShortTermMemory(max_messages=50)
        msg_default = Message.user("default importance", metadata={})
        msg_explicit = Message.user("explicit 0.5", metadata={"importance_score": 0.5})
        await mem.add_message(msg_default)
        await mem.add_message(msg_explicit)
        # Both should behave identically under forget
        forgotten = await mem.forget(threshold=0.3)
        assert forgotten == 0
        assert await mem.size() == 2

    @pytest.mark.asyncio
    async def test_importance_affects_weighted_score(self) -> None:
        """Higher importance_score produces a higher weighted score, keeping
        that message while a lower-importance one is forgotten."""
        mem = ShortTermMemory(max_messages=50)
        high = Message.user("high", metadata={"importance_score": 0.9})
        low = Message.user("low", metadata={"importance_score": 0.1})
        await mem.add_message(high)
        await mem.add_message(low)
        # Pick a threshold that retains high-importance but drops low-importance.
        # weighted ≈ time_decay * (0.8 + importance * 0.4)
        # For recent msgs time_decay ≈ 1.0
        # high: 1.0 * (0.8 + 0.9*0.4) = 1.16
        # low:  1.0 * (0.8 + 0.1*0.4) = 0.84
        forgotten = await mem.forget(threshold=0.9)
        msgs = await mem.get_context_messages()
        assert len(msgs) == 1
        assert msgs[0].content == "high"
