"""Tests for agent_harness.agent.planner — PlanAgent with MockLLM."""
from __future__ import annotations

import json

import pytest

from agent_harness.agent.planner import Plan, PlanAgent, PlanStep
from agent_harness.core.errors import MaxStepsExceededError
from agent_harness.core.message import Message, ToolCall
from agent_harness.llm.types import FinishReason, LLMResponse, Usage

from tests.conftest import MockLLM, MockTool


def _make_plan_response(goal: str, step_descriptions: list[str]) -> LLMResponse:
    """Build an LLMResponse containing a JSON plan."""
    plan_json = {
        "goal": goal,
        "steps": [
            {"id": str(i + 1), "description": desc, "status": "pending"}
            for i, desc in enumerate(step_descriptions)
        ],
    }
    return LLMResponse(
        message=Message.assistant(json.dumps(plan_json)),
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        finish_reason=FinishReason.STOP,
    )


def _make_text_response(text: str) -> LLMResponse:
    return LLMResponse(
        message=Message.assistant(text),
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        finish_reason=FinishReason.STOP,
    )


class TestPlanModel:
    def test_current_step(self) -> None:
        plan = Plan(
            goal="Test",
            steps=[
                PlanStep(id="1", description="A", status="done"),
                PlanStep(id="2", description="B", status="pending"),
            ],
        )
        assert plan.current_step is not None
        assert plan.current_step.id == "2"

    def test_current_step_none_when_complete(self) -> None:
        plan = Plan(
            goal="Test",
            steps=[PlanStep(id="1", description="A", status="done")],
        )
        assert plan.current_step is None

    def test_is_complete(self) -> None:
        plan = Plan(
            goal="G",
            steps=[
                PlanStep(id="1", description="A", status="done"),
                PlanStep(id="2", description="B", status="failed"),
            ],
        )
        assert plan.is_complete is True

    def test_is_not_complete(self) -> None:
        plan = Plan(
            goal="G",
            steps=[
                PlanStep(id="1", description="A", status="done"),
                PlanStep(id="2", description="B", status="pending"),
            ],
        )
        assert plan.is_complete is False

    def test_empty_plan_not_complete(self) -> None:
        plan = Plan(goal="G", steps=[])
        assert plan.is_complete is False

    def test_progress_summary(self) -> None:
        plan = Plan(
            goal="Research",
            steps=[
                PlanStep(id="1", description="Search", status="done", result="found 3 results"),
                PlanStep(id="2", description="Analyze", status="pending"),
            ],
        )
        summary = plan.progress_summary
        assert "Research" in summary
        assert "Search" in summary
        assert "found 3 results" in summary


class TestPlanAgentPlanGeneration:
    @pytest.mark.asyncio
    async def test_plan_generation_and_execution(self) -> None:
        """PlanAgent: generates plan → executes steps → synthesizes."""
        llm = MockLLM()

        # Response 1: plan generation
        llm.responses.append(_make_plan_response("Summarize AI", ["Search papers", "Write summary"]))
        # Response 2: execute step 1
        llm.responses.append(_make_text_response("Found 5 papers on AI safety."))
        # Response 3: execute step 2
        llm.responses.append(_make_text_response("AI safety is about alignment."))
        # Response 4: synthesis
        llm.responses.append(_make_text_response("Final: AI safety focuses on alignment."))

        agent = PlanAgent(name="planner", llm=llm, max_steps=10)
        result = await agent.run("Summarize AI safety research")

        assert result.output == "Final: AI safety focuses on alignment."
        assert result.step_count >= 3  # plan + at least 2 executions + synthesis

    @pytest.mark.asyncio
    async def test_direct_answer_without_plan(self) -> None:
        """If LLM returns plain text (not JSON plan), agent should treat it as a direct answer."""
        llm = MockLLM()
        llm.responses.append(_make_text_response("I can answer directly: 42."))

        agent = PlanAgent(name="planner", llm=llm, max_steps=5)
        result = await agent.run("Simple question")

        assert "42" in result.output


class TestPlanAgentWithTools:
    @pytest.mark.asyncio
    async def test_plan_with_tool_calls(self) -> None:
        """PlanAgent uses tools during step execution."""
        llm = MockLLM()

        # Plan
        llm.responses.append(_make_plan_response("Find data", ["Search for data"]))
        # Step execution: tool call
        llm.add_tool_call_response("mock_tool", {"query": "data search"})
        # After tool result: step completion
        llm.responses.append(_make_text_response("Step done: found data"))
        # Synthesis
        llm.responses.append(_make_text_response("Final answer with data"))

        mock_tool = MockTool(response="data found: 123")
        agent = PlanAgent(name="planner", llm=llm, tools=[mock_tool], max_steps=10)
        result = await agent.run("Find data")

        assert "data" in result.output.lower()
        assert len(mock_tool.call_history) >= 1


class TestPlanAgentMaxSteps:
    @pytest.mark.asyncio
    async def test_max_steps_exceeded(self) -> None:
        """PlanAgent should raise MaxStepsExceededError if stuck."""
        llm = MockLLM()
        # Plan with many steps
        llm.responses.append(
            _make_plan_response("Big task", [f"Step {i}" for i in range(20)])
        )
        # Never-ending tool calls for each step
        for _ in range(30):
            llm.add_tool_call_response("mock_tool", {"query": "loop"})

        mock_tool = MockTool(response="still going")
        agent = PlanAgent(name="planner", llm=llm, tools=[mock_tool], max_steps=3)

        with pytest.raises(MaxStepsExceededError):
            await agent.run("Impossible task")
