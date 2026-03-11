"""Plan-and-Execute agent: generates a plan then executes step by step."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.agent.base import BaseAgent, StepResult
from agent_harness.core.message import Message
from agent_harness.utils.json_utils import parse_json_lenient

logger = logging.getLogger(__name__)

DEFAULT_PLANNER_SYSTEM_PROMPT = """You are a planning agent. You break down complex tasks into clear, actionable steps.

When given a task:
1. First, create a plan as a JSON array of steps
2. Then execute each step one at a time
3. After each step, evaluate progress and adjust if needed

When creating a plan, respond with ONLY a JSON object in this format:
{
    "goal": "The overall goal",
    "steps": [
        {"id": "1", "description": "First step description", "status": "pending"},
        {"id": "2", "description": "Second step description", "status": "pending"}
    ]
}

When executing a step, use available tools as needed and provide the result.
When all steps are done, provide a comprehensive final answer."""


class PlanStep(BaseModel):
    """A single step in the plan."""
    id: str
    description: str
    status: str = "pending"  # pending | in_progress | done | failed
    result: str | None = None


class Plan(BaseModel):
    """A structured execution plan."""
    goal: str = ""
    steps: list[PlanStep] = Field(default_factory=list)

    @property
    def current_step(self) -> PlanStep | None:
        """Get the next pending step."""
        for step in self.steps:
            if step.status == "pending":
                return step
        return None

    @property
    def is_complete(self) -> bool:
        return all(s.status in ("done", "failed") for s in self.steps) and len(self.steps) > 0

    @property
    def progress_summary(self) -> str:
        lines = [f"Goal: {self.goal}"]
        for s in self.steps:
            marker = {"pending": "○", "in_progress": "◉", "done": "✓", "failed": "✗"}.get(s.status, "?")
            result_str = f" → {s.result[:80]}..." if s.result and len(s.result) > 80 else (f" → {s.result}" if s.result else "")
            lines.append(f"  {marker} [{s.id}] {s.description}{result_str}")
        return "\n".join(lines)


class PlanAgent(BaseAgent):
    """Plan-and-Execute agent.

    Two-phase execution:
    1. Planning: LLM generates a structured plan (JSON)
    2. Execution: Steps are executed sequentially, each as a mini ReAct loop

    Supports re-planning when execution reveals the plan needs adjustment.

    Example:
        agent = PlanAgent(
            name="researcher",
            llm=openai_provider,
            tools=[search_tool, write_tool],
        )
        result = await agent.run("Research and summarize recent AI safety papers")
    """

    def __init__(
        self,
        allow_replan: bool = True,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        if system_prompt is None:
            system_prompt = DEFAULT_PLANNER_SYSTEM_PROMPT
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.allow_replan = allow_replan
        self._plan: Plan | None = None
        self._phase: str = "planning"  # "planning" | "executing" | "synthesizing"

    async def step(self) -> StepResult:
        """Execute one step of planning or execution."""
        if self._phase == "planning":
            return await self._plan_step()
        elif self._phase == "executing":
            return await self._execute_step()
        else:  # synthesizing
            return await self._synthesize_step()

    async def _plan_step(self) -> StepResult:
        """Generate the execution plan."""
        response = await self.call_llm(tools=None)
        content = response.message.content or ""

        try:
            plan_data = parse_json_lenient(content)
            if isinstance(plan_data, dict):
                self._plan = Plan(
                    goal=plan_data.get("goal", ""),
                    steps=[PlanStep(**s) for s in plan_data.get("steps", [])],
                )
            elif isinstance(plan_data, list):
                self._plan = Plan(
                    goal="Execute task",
                    steps=[PlanStep(**s) if isinstance(s, dict) else PlanStep(id=str(i), description=str(s)) for i, s in enumerate(plan_data)],
                )
            else:
                # LLM didn't return valid plan JSON - treat as direct answer
                return StepResult(response=content)

            logger.info("PlanAgent '%s' created plan:\n%s", self.name, self._plan.progress_summary)

            # Store plan in working memory
            self.context.working_memory.set("plan", self._plan.progress_summary)
            self._phase = "executing"

            return StepResult(thought=f"Created plan with {len(self._plan.steps)} steps")

        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse plan from LLM output: %s", e)
            # If the LLM just answered directly without planning
            return StepResult(response=content)

    async def _execute_step(self) -> StepResult:
        """Execute the current plan step."""
        assert self._plan is not None

        current = self._plan.current_step
        if current is None:
            self._phase = "synthesizing"
            return await self._synthesize_step()

        current.status = "in_progress"
        self.context.working_memory.set("plan", self._plan.progress_summary)
        self.context.working_memory.set("current_step", current.description)

        # Ask LLM to execute the current step
        exec_msg = Message.user(
            f"Execute this step of the plan:\n"
            f"Step [{current.id}]: {current.description}\n\n"
            f"Current plan progress:\n{self._plan.progress_summary}\n\n"
            f"Use tools if needed. When done, provide the result for this step."
        )
        await self.context.short_term_memory.add_message(exec_msg)

        response = await self.call_llm()

        # Handle tool calls (ReAct-style within the step)
        if response.has_tool_calls and response.message.tool_calls:
            results = await self.execute_tools(response.message.tool_calls)
            current.status = "in_progress"  # still executing
            return StepResult(
                thought=f"Executing step [{current.id}]: {current.description}",
                action=response.message.tool_calls,
                observation=results,
            )

        # Step completed
        current.status = "done"
        current.result = response.message.content or "Done"
        self.context.working_memory.set("plan", self._plan.progress_summary)

        logger.info("PlanAgent step [%s] done: %s", current.id, current.result[:100] if current.result else "")

        # Check if plan is complete
        if self._plan.is_complete:
            self._phase = "synthesizing"

        return StepResult(
            thought=f"Completed step [{current.id}]",
        )

    async def _synthesize_step(self) -> StepResult:
        """Synthesize final answer from all step results."""
        assert self._plan is not None

        synthesis_msg = Message.user(
            f"All steps of the plan are now complete.\n\n"
            f"Plan summary:\n{self._plan.progress_summary}\n\n"
            f"Provide a comprehensive final answer that synthesizes all the results."
        )
        await self.context.short_term_memory.add_message(synthesis_msg)

        response = await self.call_llm(tools=None)

        return StepResult(
            thought="Synthesizing final answer",
            response=response.message.content or "",
        )
