"""Plan-and-Execute agent: generates a plan then executes step by step."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.agent.base import BaseAgent, StepResult, AgentResult
from agent_harness.core.message import Message
from agent_harness.utils.json_utils import parse_json_lenient

logger = logging.getLogger(__name__)

DEFAULT_PLANNER_SYSTEM_PROMPT = """You are a planning agent. You break down complex tasks into clear, actionable steps.

When given a task:
1. First, create a plan as a JSON object with a goal and steps
2. Then execute each step one at a time using available tools
3. After each step, evaluate progress and adjust if needed

When creating a plan, respond with ONLY a JSON object in this EXACT format:
{
    "goal": "The overall goal of the task",
    "steps": [
        {"id": "1", "description": "First step description", "status": "pending", "result": null},
        {"id": "2", "description": "Second step description", "status": "pending", "result": null}
    ]
}

Valid step status values: "pending", "in_progress", "done", "failed"
- pending: step has not been started yet
- in_progress: step is currently being executed
- done: step completed successfully (result field contains the outcome)
- failed: step failed (result field contains the error description)

IMPORTANT:
- Always respond with valid JSON when creating a plan
- Each step must have "id", "description", "status", and "result" fields
- Keep steps atomic and actionable
- When executing a step, use available tools as needed and provide the result
- When all steps are done, provide a comprehensive final answer"""


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
        """Get the next pending or in-progress step."""
        for step in self.steps:
            if step.status in ("pending", "in_progress"):
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
        max_replans: int = 2,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        if system_prompt is None:
            system_prompt = DEFAULT_PLANNER_SYSTEM_PROMPT
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.allow_replan = allow_replan
        self.max_replans = max_replans
        self._plan: Plan | None = None
        self._phase: str = "planning"  # "planning" | "executing" | "synthesizing"
        self._replan_count: int = 0

    async def run(self, input: str | Message) -> AgentResult:
        """Reset planning state before each run."""
        self._plan = None
        self._phase = "planning"
        self._replan_count = 0
        return await super().run(input)

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

        try:
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

        except Exception as e:
            logger.warning("PlanAgent step [%s] failed with error: %s", current.id, e)
            return await self._handle_step_failure(current, str(e))

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

    async def _handle_step_failure(self, step: PlanStep, error_msg: str) -> StepResult:
        """Handle a failed step — attempt replan if allowed."""
        step.status = "failed"
        step.result = error_msg
        self.context.working_memory.set("plan", self._plan.progress_summary)

        if self.allow_replan and self._replan_count < self.max_replans:
            logger.info("PlanAgent step [%s] failed, attempting replan (%d/%d)",
                        step.id, self._replan_count + 1, self.max_replans)
            return await self._do_replan()

        if self._plan.is_complete:
            self._phase = "synthesizing"

        return StepResult(thought=f"Step [{step.id}] failed: {error_msg}")

    async def _do_replan(self) -> StepResult:
        """Request LLM to update the remaining plan steps."""
        assert self._plan is not None
        self._replan_count += 1

        remaining = [s for s in self._plan.steps if s.status in ("pending",)]
        replan_msg = Message.user(
            f"The current plan needs adjustment. Here is the progress so far:\n\n"
            f"{self._plan.progress_summary}\n\n"
            f"Some steps have failed or produced unexpected results. "
            f"Please provide an updated plan for the REMAINING work only. "
            f"Keep completed steps as-is and revise or replace the {len(remaining)} "
            f"pending steps.\n\n"
            f"Respond with a JSON object in this format:\n"
            f'{{"steps": [{{"id": "...", "description": "...", "status": "pending"}}]}}'
        )
        await self.context.short_term_memory.add_message(replan_msg)

        response = await self.call_llm(tools=None)
        content = response.message.content or ""

        try:
            replan_data = parse_json_lenient(content)
            if isinstance(replan_data, dict) and "steps" in replan_data:
                new_steps = [PlanStep(**s) for s in replan_data["steps"]]
                # Replace remaining pending steps with new ones
                kept = [s for s in self._plan.steps if s.status in ("done", "failed", "in_progress")]
                self._plan.steps = kept + new_steps
                self.context.working_memory.set("plan", self._plan.progress_summary)
                logger.info("PlanAgent replanned: %d new steps", len(new_steps))
                return StepResult(thought=f"Replanned with {len(new_steps)} new steps")
        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse replan from LLM output: %s", e)

        return StepResult(thought="Replan attempted but could not parse new plan")

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
