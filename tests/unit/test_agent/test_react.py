"""Tests for agent_harness.agent.react — ReActAgent with MockLLM."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_harness.agent.base import BASE_PROMPTS
from agent_harness.agent.react import ReActAgent
from agent_harness.core.config import HarnessConfig, SkillConfig
from agent_harness.core.errors import MaxStepsExceededError
from agent_harness.core.message import Message, ToolCall
from agent_harness.llm.types import FinishReason, LLMResponse, Usage
from agent_harness.tool.builtin.skill_tool import skill_tool

from tests.conftest import MockLLM, MockTool


class TestReActSimpleResponse:
    @pytest.mark.asyncio
    async def test_single_step_text_response(self) -> None:
        """LLM returns text immediately → 1 step, no tool calls."""
        llm = MockLLM()
        llm.add_text_response("The answer is 42.")

        agent = ReActAgent(name="test", llm=llm, max_steps=5)
        result = await agent.run("What is the answer?")

        assert result.output == "The answer is 42."
        assert result.step_count == 1
        assert result.steps[0].response == "The answer is 42."
        assert result.steps[0].action is None

    @pytest.mark.asyncio
    async def test_llm_receives_system_and_user_messages(self) -> None:
        llm = MockLLM()
        llm.add_text_response("Done")

        agent = ReActAgent(name="test", llm=llm, max_steps=5)
        await agent.run("Hello")

        # call_history[0] should include system prompt + user message
        messages = llm.call_history[0]
        roles = [m.role.value for m in messages]
        assert "system" in roles
        assert "user" in roles


class TestReActToolUsage:
    @pytest.mark.asyncio
    async def test_tool_call_then_response(self) -> None:
        """LLM calls a tool, then provides final answer → 2 steps."""
        llm = MockLLM()
        llm.add_tool_call_response("mock_tool", {"query": "search this"})
        llm.add_text_response("Based on the search: the answer is X.")

        mock_tool = MockTool(response="search result: X")
        agent = ReActAgent(name="test", llm=llm, tools=[mock_tool], max_steps=5)
        result = await agent.run("Find X")

        assert result.step_count == 2
        # Step 1: tool call
        assert result.steps[0].action is not None
        assert len(result.steps[0].action) == 1
        assert result.steps[0].action[0].name == "mock_tool"
        assert result.steps[0].observation is not None
        assert result.steps[0].observation[0].content == "search result: X"
        assert result.steps[0].response is None

        # Step 2: final answer
        assert result.steps[1].response is not None
        assert "answer is X" in result.steps[1].response
        assert result.output == "Based on the search: the answer is X."

        # Tool was actually called
        assert len(mock_tool.call_history) == 1
        assert mock_tool.call_history[0] == {"query": "search this"}


class TestReActMaxSteps:
    @pytest.mark.asyncio
    async def test_max_steps_exceeded(self) -> None:
        """LLM always calls tools → exceeds max_steps."""
        llm = MockLLM()
        # Queue more tool calls than max_steps
        for _ in range(10):
            llm.add_tool_call_response("mock_tool", {"query": "again"})

        mock_tool = MockTool(response="still searching")
        agent = ReActAgent(name="test", llm=llm, tools=[mock_tool], max_steps=3)

        with pytest.raises(MaxStepsExceededError):
            await agent.run("Never-ending search")

    @pytest.mark.asyncio
    async def test_finishes_at_exactly_max_steps(self) -> None:
        """LLM uses tools for (max_steps - 1) times, then responds at the limit."""
        llm = MockLLM()
        llm.add_tool_call_response("mock_tool", {"query": "q1"})
        llm.add_tool_call_response("mock_tool", {"query": "q2"})
        llm.add_text_response("Final answer at step 3")

        mock_tool = MockTool(response="result")
        agent = ReActAgent(name="test", llm=llm, tools=[mock_tool], max_steps=3)
        result = await agent.run("Complex question")

        assert result.step_count == 3
        assert result.output == "Final answer at step 3"


@pytest.fixture
def restore_harness_config() -> Iterator[None]:
    original = HarnessConfig._instance
    try:
        yield
    finally:
        HarnessConfig._instance = original


@pytest.fixture(autouse=True)
def reset_skill_tool_cache() -> Iterator[None]:
    skill_tool._loader = None
    skill_tool._loader_dirs_key = None
    skill_tool._loader_state_key = None
    skill_tool._state_key_checked_at = 0.0
    try:
        yield
    finally:
        skill_tool._loader = None
        skill_tool._loader_dirs_key = None
        skill_tool._loader_state_key = None
        skill_tool._state_key_checked_at = 0.0


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    sd = root / "demo"
    sd.mkdir(parents=True)
    (sd / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill.\n---\n\nDemo body.\n"
    )
    return root


class TestSkillPromptSupplement:
    def test_supplement_appended_when_skill_tool_present(
        self,
        skills_dir: Path,
        restore_harness_config: None,
    ) -> None:
        HarnessConfig._instance = HarnessConfig(skill=SkillConfig(dirs=[str(skills_dir)]))
        agent = ReActAgent(name="test", tools=[skill_tool])
        assert "## Skills" in agent.system_prompt
        assert "skill_tool" in agent.system_prompt

    def test_no_supplement_without_skill_tool(self) -> None:
        agent = ReActAgent(name="test", tools=[])
        assert "## Skills" not in agent.system_prompt

    def test_custom_prompt_gets_supplement(
        self,
        skills_dir: Path,
        restore_harness_config: None,
    ) -> None:
        HarnessConfig._instance = HarnessConfig(skill=SkillConfig(dirs=[str(skills_dir)]))
        agent = ReActAgent(
            name="test",
            system_prompt="Custom prompt.",
            tools=[skill_tool],
        )
        assert agent.system_prompt.startswith("Custom prompt.")
        assert "## Skills" in agent.system_prompt

    def test_custom_prompt_no_supplement_without_skill_tool(self) -> None:
        agent = ReActAgent(name="test", system_prompt="Custom prompt.", tools=[])
        assert agent.system_prompt == "Custom prompt."

    async def test_should_inject_consistent_with_supplement(
        self,
        skills_dir: Path,
        restore_harness_config: None,
    ) -> None:
        HarnessConfig._instance = HarnessConfig(skill=SkillConfig(dirs=[str(skills_dir)]))
        agent = ReActAgent(name="test", tools=[skill_tool])

        assert await agent._should_inject_system_prompt() is True

        await agent.context.short_term_memory.add_message(
            Message.system(agent.system_prompt)
        )

        assert await agent._should_inject_system_prompt() is False
