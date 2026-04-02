"""Tests for agent_harness.memory.short_term — ShortTermMemory buffer and trimming."""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from agent_harness.core.message import Message, Role
from agent_harness.memory.compressor import ContextCompressor
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
        assert all("msg-" in item.content for item in items)

    @pytest.mark.asyncio
    async def test_add_message_does_not_trim(self) -> None:
        """add_message only appends — no trimming until get_context_messages."""
        mem = ShortTermMemory(max_tokens=50)
        for i in range(20):
            await mem.add_message(Message.user(f"message-{i} " * 10))
        assert await mem.size() == 20


class TestTokenTrim:
    @pytest.mark.asyncio
    async def test_trim_on_get_context(self) -> None:
        """Token trimming happens in get_context_messages, not add_message."""
        mem = ShortTermMemory(max_tokens=50)
        for i in range(20):
            await mem.add_message(Message.user(f"message-{i} " * 10))
        msgs = await mem.get_context_messages()
        assert len(msgs) < 20

    @pytest.mark.asyncio
    async def test_system_message_preserved(self) -> None:
        mem = ShortTermMemory(max_tokens=100)
        await mem.add_message(Message.system("System prompt"))
        for i in range(20):
            await mem.add_message(Message.user(f"msg-{i} " * 10))
        msgs = await mem.get_context_messages()
        assert msgs[0].role == Role.SYSTEM
        assert msgs[0].content == "System prompt"

    @pytest.mark.asyncio
    async def test_no_trim_when_under_budget(self) -> None:
        mem = ShortTermMemory(max_tokens=100000)
        await mem.add_message(Message.user("one"))
        await mem.add_message(Message.user("two"))
        msgs = await mem.get_context_messages()
        assert len(msgs) == 2

    @pytest.mark.asyncio
    async def test_multiple_system_messages(self) -> None:
        mem = ShortTermMemory(max_tokens=200)
        await mem.add_message(Message.system("sys1"))
        await mem.add_message(Message.system("sys2"))
        for i in range(20):
            await mem.add_message(Message.user(f"u{i} " * 10))
        msgs = await mem.get_context_messages()
        sys_msgs = [m for m in msgs if m.role == Role.SYSTEM]
        assert len(sys_msgs) == 2

    @pytest.mark.asyncio
    async def test_keeps_most_recent(self) -> None:
        mem = ShortTermMemory(max_tokens=100)
        for i in range(20):
            await mem.add_message(Message.user(f"msg-{i} " * 10))
        msgs = await mem.get_context_messages()
        contents = [m.content for m in msgs]
        assert "msg-19 " * 10 in contents[-1]


class TestCompressorAttribute:
    @pytest.mark.asyncio
    async def test_compressor_default_none(self) -> None:
        mem = ShortTermMemory()
        assert mem.compressor is None

    @pytest.mark.asyncio
    async def test_compressor_is_public(self) -> None:
        from unittest.mock import AsyncMock

        from agent_harness.memory.compressor import ContextCompressor

        compressor = ContextCompressor(
            llm=AsyncMock(), threshold=0.75, retain_count=4, model="gpt-4o",
        )
        mem = ShortTermMemory(max_tokens=500, compressor=compressor)
        assert mem.compressor is compressor

    @pytest.mark.asyncio
    async def test_compression_failure_logs_warning_and_falls_back_to_trim(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        compressor = ContextCompressor(
            llm=AsyncMock(),
            threshold=0.0,
            retain_count=4,
            model="gpt-4o",
        )

        mem = ShortTermMemory(max_tokens=50, compressor=compressor)
        for i in range(20):
            await mem.add_message(Message.user(f"message-{i} " * 10))

        with (
            patch.object(
                compressor,
                "compress",
                AsyncMock(side_effect=RuntimeError("boom")),
            ),
            caplog.at_level(logging.WARNING),
        ):
            msgs = await mem.get_context_messages()

        assert len(msgs) > 0
        assert "Context compression failed" in caplog.text


class TestForgetImportanceScore:
    """Tests for forget() using importance_score from message metadata."""

    @pytest.mark.asyncio
    async def test_high_importance_retained(self) -> None:
        mem = ShortTermMemory()
        msg = Message.user("important", metadata={"importance_score": 0.9})
        await mem.add_message(msg)
        forgotten = await mem.forget(threshold=0.3)
        assert forgotten == 0
        assert await mem.size() == 1

    @pytest.mark.asyncio
    async def test_low_importance_forgotten(self) -> None:
        mem = ShortTermMemory()
        msg = Message.user("trivial", metadata={"importance_score": 0.1})
        await mem.add_message(msg)
        forgotten = await mem.forget(threshold=1.1)
        assert forgotten == 1
        assert await mem.size() == 0

    @pytest.mark.asyncio
    async def test_default_importance_when_missing(self) -> None:
        mem = ShortTermMemory()
        msg_default = Message.user("default importance", metadata={})
        msg_explicit = Message.user("explicit 0.5", metadata={"importance_score": 0.5})
        await mem.add_message(msg_default)
        await mem.add_message(msg_explicit)
        forgotten = await mem.forget(threshold=0.3)
        assert forgotten == 0
        assert await mem.size() == 2

    @pytest.mark.asyncio
    async def test_importance_affects_weighted_score(self) -> None:
        mem = ShortTermMemory()
        high = Message.user("high", metadata={"importance_score": 0.9})
        low = Message.user("low", metadata={"importance_score": 0.1})
        await mem.add_message(high)
        await mem.add_message(low)
        await mem.forget(threshold=0.9)
        msgs = await mem.get_context_messages()
        assert len(msgs) == 1
        assert msgs[0].content == "high"
