"""Plan-and-Execute agent: multi-agent orchestration for plan -> execute -> replan."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.agent.base import BaseAgent, StepResult, AgentResult
from agent_harness.context.state import AgentState
from agent_harness.core.message import Message
from agent_harness.llm.types import Usage
from agent_harness.utils.json_utils import parse_json_lenient
from agent_harness.utils.token_counter import truncate_text_by_tokens

logger = logging.getLogger(__name__)

_PLAN_RESULT_MAX_TOKENS = 26
_PLAN_EVENT_OUTPUT_MAX_TOKENS = 64


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

PlanAndExecutePrompts: dict[str, str] = {
    "planner": (
        "You are a planning agent specialized in task decomposition and strategic planning.\n"
        "\n"
        "## Role\n"
        "Analyze the user's request and break it down into a structured, actionable plan.\n"
        "Think carefully about dependencies between steps and the optimal execution order.\n"
        "\n"
        "## Output Format\n"
        "Respond with ONLY a valid JSON object. Do not include any text, markdown, or explanation\n"
        "outside the JSON structure.\n"
        "\n"
        "Required schema:\n"
        "{\n"
        '    "goal": "A clear, refined statement of the overall objective",\n'
        '    "steps": [\n'
        '        {\n'
        '            "id": "1",\n'
        '            "description": "Clear description of what this step accomplishes"\n'
        '        }\n'
        "    ]\n"
        "}\n"
        "\n"
        "## Planning Rules\n"
        "1. Keep each step atomic — one clear action per step\n"
        "2. Steps must be independently executable given prior step results\n"
        "3. Order steps by dependency: prerequisites before dependents\n"
        "4. Aim for 2-6 steps; avoid over-decomposition for simple tasks\n"
        "5. Each step should produce a verifiable outcome\n"
        "\n"
        "## Constraints\n"
        "- Output ONLY valid JSON — no markdown fences, no preamble, no commentary\n"
        "- Every step MUST have 'id' and 'description' fields\n"
        "- The 'id' field must be a unique string (e.g., '1', '2', '3')\n"
        "- Do not reference tools or implementation details — focus on WHAT, not HOW\n"
        "- If the task is trivial, a single-step plan is acceptable"
    ),
    "executor": (
        "You are an execution agent responsible for completing a specific step of a plan.\n"
        "\n"
        "## Role\n"
        "You receive a single step to execute along with context from previously completed steps.\n"
        "Use the available tools to gather information, perform actions, and accomplish the step.\n"
        "\n"
        "## Execution Rules\n"
        "1. Focus ONLY on the current step — do not attempt other steps\n"
        "2. Use tools when you need external information or must perform actions\n"
        "3. You may call multiple tools in sequence if the step requires it\n"
        "4. Analyze tool results before deciding next actions\n"
        "5. When the step is complete, provide a clear, concise summary of the result\n"
        "\n"
        "## Output Guidelines\n"
        "- Your final response should summarize what was accomplished\n"
        "- Include key data, findings, or outcomes from tool usage\n"
        "- Report failures clearly if the step could not be completed\n"
        "- Keep the result focused and actionable for subsequent steps\n"
        "\n"
        "## Constraints\n"
        "- Do not attempt steps outside your current assignment\n"
        "- Do not modify the plan — only execute the assigned step\n"
        "- If a step cannot be completed with available tools, explain why\n"
        "- Provide your result as plain text (not JSON)"
    ),
    "replanner": (
        "You are a replanning agent that evaluates execution progress and decides the next action.\n"
        "\n"
        "## Role\n"
        "After each step is executed, you receive the original goal, the current plan status,\n"
        "and the result of the latest step. Your job is to evaluate whether the goal has been\n"
        "achieved, whether the remaining plan is still valid, or whether replanning is needed.\n"
        "\n"
        "## Output Format\n"
        "Respond with ONLY a valid JSON object in one of these three formats:\n"
        "\n"
        "### Goal Achieved — task is complete\n"
        '{"goal_achieved": true, "final_answer": "Comprehensive answer addressing the original task"}\n'
        "\n"
        "### Continue — proceed with the next step as planned\n"
        '{"goal_achieved": false, "should_replan": false}\n'
        "\n"
        "### Replan — modify the remaining steps\n"
        "{\n"
        '    "goal_achieved": false,\n'
        '    "should_replan": true,\n'
        '    "reason": "Explanation of why replanning is needed",\n'
        '    "updated_steps": [\n'
        '        {"id": "N", "description": "New or revised step"}\n'
        "    ]\n"
        "}\n"
        "\n"
        "## Evaluation Rules\n"
        "1. Set goal_achieved=true ONLY when you have enough information for a complete,\n"
        "   comprehensive answer to the original task\n"
        "2. The final_answer must directly and fully address the original task\n"
        "3. Use should_replan=true when step results reveal the plan is insufficient,\n"
        "   incorrect, or needs adjustment\n"
        "4. updated_steps replaces ALL remaining pending steps (completed steps are preserved)\n"
        "5. If the step failed but remaining steps can still achieve the goal, continue\n"
        "\n"
        "## Constraints\n"
        "- Output ONLY valid JSON — no markdown fences, no preamble, no commentary\n"
        "- Do not add steps unnecessarily — only replan when genuinely needed\n"
        "- The final_answer should be self-contained and comprehensive\n"
        "- When replanning, ensure updated_steps are actionable and ordered correctly"
    ),
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    """A single step in the plan."""

    id: str
    description: str
    status: str = "pending"
    result: str | None = None


class Plan(BaseModel):
    """A structured execution plan."""

    goal: str = ""
    steps: list[PlanStep] = Field(default_factory=list)

    @property
    def current_step(self) -> PlanStep | None:
        """Return the next pending or in-progress step."""
        for step in self.steps:
            if step.status in ("pending", "in_progress"):
                return step
        return None

    @property
    def is_complete(self) -> bool:
        """True when all steps are done or failed."""
        return (
            all(s.status in ("done", "failed") for s in self.steps)
            and len(self.steps) > 0
        )

    @property
    def progress_summary(self) -> str:
        """Human-readable progress summary."""
        lines = [f"Goal: {self.goal}"]
        for s in self.steps:
            marker = {
                "pending": "○",
                "in_progress": "◉",
                "done": "✓",
                "failed": "✗",
            }.get(s.status, "?")
            if s.result:
                result_str = " -> " + truncate_text_by_tokens(
                    s.result,
                    max_tokens=_PLAN_RESULT_MAX_TOKENS,
                    suffix="...",
                )
            else:
                result_str = ""
            lines.append(f"  {marker} [{s.id}] {s.description}{result_str}")
        return "\n".join(lines)


class ReplanDecision(BaseModel):
    """Structured decision from the replanner agent."""

    goal_achieved: bool = False
    should_replan: bool = False
    reason: str = ""
    updated_steps: list[PlanStep] | None = None
    final_answer: str | None = None


# ---------------------------------------------------------------------------
# Sub-agents
# ---------------------------------------------------------------------------


class PlannerAgent(BaseAgent):
    """Generates a structured plan from the user's request.

    Pure LLM reasoning — no tools. Produces a JSON plan matching the
    Plan / PlanStep schema. Always completes in a single step.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("system_prompt", PlanAndExecutePrompts["planner"])
        super().__init__(**kwargs)

    async def step(self) -> StepResult:
        response = await self.call_llm(tools=None)
        return StepResult(response=response.message.content)


class ExecutorAgent(BaseAgent):
    """Executes a single plan step using available tools (ReAct-style).

    Supports multi-step tool calling via the standard BaseAgent.run() loop.
    Each call to step() either invokes tools (loop continues) or returns
    a final text result (loop ends).
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("system_prompt", PlanAndExecutePrompts["executor"])
        super().__init__(**kwargs)

    async def step(self) -> StepResult:
        response = await self.call_llm()
        if response.has_tool_calls:
            tool_calls = response.message.tool_calls or []
            results = await self.execute_tools(tool_calls)
            return StepResult(
                thought=response.message.content,
                action=tool_calls,
                observation=results,
            )
        return StepResult(response=response.message.content or "")


class ReplannerAgent(BaseAgent):
    """Evaluates step results and decides whether to continue, replan, or finish.

    Pure LLM reasoning — no tools. Returns a JSON decision that the
    orchestrator parses into a ReplanDecision. Single-step execution.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("system_prompt", PlanAndExecutePrompts["replanner"])
        super().__init__(**kwargs)

    async def step(self) -> StepResult:
        response = await self.call_llm(tools=None)
        return StepResult(response=response.message.content)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class PlanAndExecuteAgent(BaseAgent):
    """Multi-agent Plan-and-Execute orchestrator.

    Coordinates three sub-agents (Planner, Executor, Replanner) through
    a plan-execute-evaluate loop. Each sub-agent runs in a forked context
    for memory isolation.

    Workflow:
        1. PlannerAgent generates a structured plan
        2. ExecutorAgent executes steps one by one (with tool access)
        3. ReplannerAgent evaluates results and adjusts the plan
        4. Loop until goal achieved or max iterations reached
    """

    def __init__(
        self,
        max_replans: int = 3,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(system_prompt=system_prompt or "", **kwargs)
        self.max_replans = max_replans

    async def step(self) -> StepResult:
        """Not used — run() is overridden."""
        return StepResult()

    async def run(self, input: str | Message) -> AgentResult:
        if self.context.state.is_terminal:
            self.context.state.reset()

        input_text = input if isinstance(input, str) else (input.content or "")

        self.context.state.transition(AgentState.THINKING)
        await self.hooks.on_run_start(self.name, input_text)
        await self.emit("agent.run.start", agent=self.name, input=input_text)

        total_usage = Usage()
        steps: list[StepResult] = []

        try:
            planner = PlannerAgent(
                name=f"{self.name}.planner",
                llm=self.llm,
                tools=[],
                context=self.context.fork("planner"),
                hooks=self.hooks,
                max_steps=1,
            )
            executor = ExecutorAgent(
                name=f"{self.name}.executor",
                llm=self.llm,
                tools=self.tools,
                context=self.context.fork("executor"),
                hooks=self.hooks,
                max_steps=self.max_steps,
            )
            replanner = ReplannerAgent(
                name=f"{self.name}.replanner",
                llm=self.llm,
                tools=[],
                context=self.context.fork("replanner"),
                hooks=self.hooks,
                max_steps=1,
            )

            # Planning phase
            plan_result = await planner.run(input_text)
            total_usage = total_usage + plan_result.usage
            plan = self._parse_plan(plan_result.output, input_text)

            await self.emit(
                "agent.plan.created",
                agent=self.name,
                plan=plan.progress_summary,
            )

            # Execute-Evaluate loop
            final_output = ""
            iteration = 0

            while iteration < self.max_steps:
                iteration += 1

                if plan.is_complete:
                    break

                current = plan.current_step
                if current is None:
                    break

                current.status = "in_progress"
                self.context.state.transition(AgentState.ACTING)

                executor.context.working_memory.set("plan", plan.progress_summary)
                executor.context.working_memory.set(
                    "current_step", current.description
                )

                exec_result = await executor.run(current.description)
                total_usage = total_usage + exec_result.usage

                current.status = "done"
                current.result = exec_result.output

                steps.append(
                    StepResult(thought=f"Executed step [{current.id}]")
                )

                self.context.state.transition(AgentState.OBSERVING)
                self.context.state.transition(AgentState.THINKING)

                replanner.context.working_memory.set("goal", plan.goal)
                replanner.context.working_memory.set(
                    "plan", plan.progress_summary
                )
                replanner.context.working_memory.set(
                    "current_step",
                    f"Step [{current.id}] result: {current.result}",
                )

                replan_result = await replanner.run(input_text)
                total_usage = total_usage + replan_result.usage

                decision = self._parse_replan_decision(replan_result.output)
                await self.emit("agent.replan.decision", agent=self.name)

                if decision.goal_achieved:
                    final_output = decision.final_answer or exec_result.output
                    break

                if decision.should_replan and decision.updated_steps:
                    kept = [
                        s
                        for s in plan.steps
                        if s.status in ("done", "failed")
                    ]
                    plan.steps = kept + decision.updated_steps

            else:
                final_output = (
                    f"Max iterations reached ({self.max_steps}). "
                    f"Partial result: {plan.progress_summary}"
                )

            if not final_output:
                final_output = plan.progress_summary

            self.context.state.transition(AgentState.FINISHED)
            messages = await self.context.short_term_memory.get_context_messages()

            result = AgentResult(
                output=final_output,
                messages=messages,
                steps=steps,
                usage=total_usage,
            )

            await self.hooks.on_run_end(self.name, final_output)
            await self.emit(
                "agent.run.end",
                agent=self.name,
                output=truncate_text_by_tokens(
                    final_output,
                    max_tokens=_PLAN_EVENT_OUTPUT_MAX_TOKENS,
                    suffix="",
                ),
            )
            return result

        except Exception as e:
            await self.hooks.on_error(self.name, e)
            await self.emit(
                "agent.run.error", agent=self.name, error=str(e)
            )
            if not self.context.state.is_terminal:
                self.context.state.transition(AgentState.ERROR)
            raise

    @staticmethod
    def _parse_plan(content: str, fallback_goal: str) -> Plan:
        """Parse planner output into a Plan object, with fallback."""
        try:
            data = parse_json_lenient(content)
            if isinstance(data, dict):
                raw_steps = data.get("steps", [])
                parsed_steps = [
                    PlanStep(**s) if isinstance(s, dict)
                    else PlanStep(id=str(i), description=str(s))
                    for i, s in enumerate(raw_steps)
                ]
                return Plan(
                    goal=data.get("goal", fallback_goal), steps=parsed_steps
                )
        except (ValueError, TypeError):
            pass
        return Plan(
            goal=fallback_goal,
            steps=[PlanStep(id="1", description=fallback_goal)],
        )

    @staticmethod
    def _parse_replan_decision(content: str) -> ReplanDecision:
        """Parse replanner output into a ReplanDecision."""
        try:
            data = parse_json_lenient(content)
            if isinstance(data, dict):
                updated = None
                if data.get("updated_steps"):
                    updated = [
                        PlanStep(**s) if isinstance(s, dict)
                        else PlanStep(id=str(i), description=str(s))
                        for i, s in enumerate(data["updated_steps"])
                    ]
                return ReplanDecision(
                    goal_achieved=data.get("goal_achieved", False),
                    should_replan=data.get("should_replan", False),
                    reason=data.get("reason", ""),
                    updated_steps=updated,
                    final_answer=data.get("final_answer"),
                )
        except (ValueError, TypeError):
            pass
        return ReplanDecision()


# Backward compatibility alias
PlanAgent = PlanAndExecuteAgent
