"""Tests for agent_harness.memory.working — WorkingMemory scratchpad and history."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from agent_harness.core.message import Message, Role
from agent_harness.memory.working_term import WorkingMemory, ScratchpadCategory
from agent_harness.utils.token_counter import count_tokens


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
        assert "### Objective" in result
        assert "find data" in result
        assert "### Current Progress" in result
        assert "status" in result

    def test_long_value_truncated(self) -> None:
        wm = WorkingMemory()
        long_text = "token " * 8000
        wm.set("big", long_text)
        result = wm.to_prompt_string()
        formatted = WorkingMemory._format_value(long_text)
        assert formatted.endswith("...")
        assert formatted in result
        assert count_tokens(formatted) <= 5000

    def test_goal_auto_categorizes_under_objective(self) -> None:
        wm = WorkingMemory()
        wm.set("goal", "win the race")
        result = wm.to_prompt_string()
        assert "### Objective" in result
        assert "**goal**:" in result
        assert "win the race" in result

    def test_steps_auto_categorizes_under_plan(self) -> None:
        wm = WorkingMemory()
        wm.set("steps", "step1, step2")
        wm.set("plan", "master plan")
        result = wm.to_prompt_string()
        assert "### Plan" in result
        assert "**steps**:" in result
        assert "step1, step2" in result
        assert "**plan**:" in result
        assert "master plan" in result

    def test_current_task_auto_categorizes_under_progress(self) -> None:
        wm = WorkingMemory()
        wm.set("current_task", "doing stuff")
        result = wm.to_prompt_string()
        assert "### Current Progress" in result
        assert "**current_task**:" in result
        assert "doing stuff" in result

    def test_unknown_keys_go_to_additional_context(self) -> None:
        wm = WorkingMemory()
        wm.set("custom_key", "custom_value")
        result = wm.to_prompt_string()
        assert "### Additional Context" in result
        assert "**custom_key**:" in result
        assert "custom_value" in result

    def test_set_with_category_override(self) -> None:
        wm = WorkingMemory()
        # "custom_key" would normally be uncategorized; override to plan
        wm.set("custom_key", "my plan detail", category="plan")
        result = wm.to_prompt_string()
        assert "### Plan" in result
        assert "**custom_key**:" in result
        assert "my plan detail" in result
        assert "### Additional Context" not in result

    def test_history_under_key_observations(self) -> None:
        wm = WorkingMemory()
        wm._history.append(Message.user("saw something", metadata={"importance_score": 0.5}))
        wm._history.append(Message.user("found a clue", metadata={"importance_score": 0.8}))
        result = wm.to_prompt_string()
        assert "### Key Observations" in result
        assert "- saw something" in result
        assert "- found a clue" in result

    def test_category_ordering(self) -> None:
        wm = WorkingMemory()
        wm.set("errors", "oops")
        wm.set("goal", "win")
        wm.set("current_step", "step 3")
        wm.set("plan", "do things")
        result = wm.to_prompt_string()
        # Enum order: OBJECTIVE, PLAN, PROGRESS, OBSERVATION, ERROR
        obj_pos = result.index("### Objective")
        plan_pos = result.index("### Plan")
        prog_pos = result.index("### Current Progress")
        err_pos = result.index("### Errors")
        assert obj_pos < plan_pos < prog_pos < err_pos


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
        assert "Plan" in msgs[0].content

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


class TestForgetTimeDecay:
    """Tests for the time-decay + importance scoring forget() method."""

    @pytest.mark.asyncio
    async def test_recent_high_importance_kept(self) -> None:
        wm = WorkingMemory()
        await wm.add("important recent", metadata={"importance_score": 1.0})
        removed = await wm.forget(threshold=0.3)
        assert removed == 0
        assert len(wm._history) == 1
        assert wm._history[0].content == "important recent"

    @pytest.mark.asyncio
    async def test_old_low_importance_forgotten(self) -> None:
        wm = WorkingMemory()
        # Manually insert an old entry with low importance
        old_time = datetime.now() - timedelta(hours=500)
        wm._history.append(Message.user("stale note", metadata={"importance_score": 0.1}, created_at=old_time))
        removed = await wm.forget(threshold=0.3)
        assert removed == 1
        assert len(wm._history) == 0

    @pytest.mark.asyncio
    async def test_scratchpad_unaffected_by_forget(self) -> None:
        wm = WorkingMemory()
        wm.set("plan", "step1")
        wm.set("goal", "win")
        # Add an old entry that will be forgotten
        old_time = datetime.now() - timedelta(hours=500)
        wm._history.append(Message.user("old note", metadata={"importance_score": 0.0}, created_at=old_time))
        removed = await wm.forget(threshold=0.3)
        assert removed == 1
        # Scratchpad keys are always preserved
        assert wm.get("plan") == "step1"
        assert wm.get("goal") == "win"

    @pytest.mark.asyncio
    async def test_forget_empty_history(self) -> None:
        wm = WorkingMemory()
        removed = await wm.forget()
        assert removed == 0

    @pytest.mark.asyncio
    async def test_mixed_entries(self) -> None:
        wm = WorkingMemory()
        # Recent high importance — kept
        await wm.add("fresh", metadata={"importance_score": 0.9})
        # Old low importance — forgotten
        old_time = datetime.now() - timedelta(hours=500)
        wm._history.append(Message.user("ancient", metadata={"importance_score": 0.1}, created_at=old_time))
        # Recent default importance — kept
        await wm.add("recent default")

        removed = await wm.forget(threshold=0.3)
        assert removed == 1
        contents = [msg.content for msg in wm._history]
        assert "fresh" in contents
        assert "recent default" in contents
        assert "ancient" not in contents
