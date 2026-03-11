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
    async def test_query_returns_recent(self) -> None:
        mem = ShortTermMemory()
        for i in range(10):
            await mem.add_message(Message.user(f"msg-{i}"))
        items = await mem.query("anything", top_k=3)
        assert len(items) == 3
        assert items[-1].content == "msg-9"


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
