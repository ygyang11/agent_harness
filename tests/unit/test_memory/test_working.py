"""Tests for agent_harness.memory.working — WorkingMemory scratchpad and history."""
from __future__ import annotations

import pytest

from agent_harness.core.message import Message, Role
from agent_harness.memory.working import WorkingMemory


class TestWorkingMemoryKV:
    def test_set_and_get(self) -> None:
        wm = WorkingMemory()
        wm.set("key1", "value1")
        assert wm.get("key1") == "value1"

    def test_get_default(self) -> None:
        wm = WorkingMemory()
        assert wm.get("missing") is None
        assert wm.get("missing", "fallback") == "fallback"

    def test_delete(self) -> None:
        wm = WorkingMemory()
        wm.set("k", "v")
        wm.delete("k")
        assert wm.get("k") is None

    def test_delete_nonexistent(self) -> None:
        wm = WorkingMemory()
        wm.delete("nope")  # should not raise

    def test_keys(self) -> None:
        wm = WorkingMemory()
        wm.set("a", 1)
        wm.set("b", 2)
        assert sorted(wm.keys()) == ["a", "b"]

    def test_to_dict(self) -> None:
        wm = WorkingMemory()
        wm.set("x", 10)
        wm.set("y", 20)
        d = wm.to_dict()
        assert d == {"x": 10, "y": 20}
        # Returned dict is a copy
        d["z"] = 30
        assert wm.get("z") is None

    def test_overwrite(self) -> None:
        wm = WorkingMemory()
        wm.set("k", "old")
        wm.set("k", "new")
        assert wm.get("k") == "new"


class TestToPromptString:
    def test_empty_scratchpad(self) -> None:
        wm = WorkingMemory()
        assert wm.to_prompt_string() == ""

    def test_non_empty_scratchpad(self) -> None:
        wm = WorkingMemory()
        wm.set("goal", "find data")
        wm.set("status", "in progress")
        result = wm.to_prompt_string()
        assert "## Working Memory" in result
        assert "find data" in result
        assert "status" in result

    def test_long_value_truncated(self) -> None:
        wm = WorkingMemory()
        wm.set("big", "x" * 600)
        result = wm.to_prompt_string()
        assert "..." in result


class TestWorkingMemoryBaseMemory:
    @pytest.mark.asyncio
    async def test_add(self) -> None:
        wm = WorkingMemory()
        await wm.add("first note")
        await wm.add("second note", metadata={"key": "val"})
        items = await wm.query("anything", top_k=10)
        assert len(items) == 2
        assert items[0].content == "first note"
        # metadata should be stored in scratchpad
        assert wm.get("key") == "val"

    @pytest.mark.asyncio
    async def test_add_message(self) -> None:
        wm = WorkingMemory()
        await wm.add_message(Message.user("user said this"))
        items = await wm.query("x", top_k=5)
        assert len(items) == 1
        assert items[0].content == "user said this"

    @pytest.mark.asyncio
    async def test_add_message_no_content(self) -> None:
        wm = WorkingMemory()
        msg = Message(role=Role.ASSISTANT, content=None)
        await wm.add_message(msg)
        items = await wm.query("x", top_k=5)
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_query_top_k(self) -> None:
        wm = WorkingMemory()
        for i in range(10):
            await wm.add(f"note-{i}")
        items = await wm.query("note", top_k=3)
        assert len(items) == 3
        assert all("note-" in item.content for item in items)

    @pytest.mark.asyncio
    async def test_get_context_messages_empty(self) -> None:
        wm = WorkingMemory()
        msgs = await wm.get_context_messages()
        assert msgs == []

    @pytest.mark.asyncio
    async def test_get_context_messages_with_data(self) -> None:
        wm = WorkingMemory()
        wm.set("plan", "step1, step2")
        msgs = await wm.get_context_messages()
        assert len(msgs) == 1
        assert msgs[0].role == Role.SYSTEM
        assert "Working Memory" in msgs[0].content

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        wm = WorkingMemory()
        wm.set("a", 1)
        await wm.add("note")
        await wm.clear()
        assert wm.get("a") is None
        assert wm.keys() == []
        items = await wm.query("x", top_k=10)
        assert items == []
        assert await wm.size() == 0

    @pytest.mark.asyncio
    async def test_size(self) -> None:
        wm = WorkingMemory()
        assert await wm.size() == 0
        wm.set("k1", "v1")
        await wm.add("note")
        # size = scratchpad entries + history entries
        assert await wm.size() == 2
