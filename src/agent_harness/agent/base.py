"""Base agent class for agent_harness."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.core.errors import AgentError, MaxStepsExceededError
from agent_harness.core.event import EventEmitter
from agent_harness.core.message import Message, ToolCall, ToolResult
from agent_harness.llm.base import BaseLLM
from agent_harness.llm.types import LLMResponse, Usage
from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.tool.executor import ToolExecutor
from agent_harness.tool.registry import ToolRegistry
from agent_harness.context.context import AgentContext
from agent_harness.context.state import AgentState
from agent_harness.agent.hooks import AgentHooks, DefaultHooks

logger = logging.getLogger(__name__)


class StepResult(BaseModel):
    """Result of a single agent step."""
    thought: str | None = None
    action: list[ToolCall] | None = None
    observation: list[ToolResult] | None = None
    response: str | None = None  # final response if step produced one


class AgentResult(BaseModel):
    """Final result of an agent run."""
    output: str
    messages: list[Message] = Field(default_factory=list)
    steps: list[StepResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

    @property
    def step_count(self) -> int:
        return len(self.steps)


class BaseAgent(ABC, EventEmitter):
    """Abstract base class for all agents.

    Provides the run loop, tool execution, and lifecycle management.
    Subclasses implement step() to define their reasoning strategy.

    Args:
        name: Unique agent name.
        llm: LLM provider for generation.
        tools: List of available tools.
        context: Agent runtime context.
        hooks: Lifecycle hooks for observation/modification.
        max_steps: Maximum steps before forced termination.
        system_prompt: System prompt for the agent.
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        tools: list[BaseTool] | None = None,
        context: AgentContext | None = None,
        hooks: AgentHooks | None = None,
        max_steps: int = 20,
        system_prompt: str = "",
    ) -> None:
        self.name = name
        self.llm = llm
        self.context = context or AgentContext.create()
        self.hooks = hooks or DefaultHooks()
        self.max_steps = max_steps
        self.system_prompt = system_prompt

        # Set up tool registry and executor
        self.tool_registry = ToolRegistry()
        for t in (tools or []):
            self.tool_registry.register(t)
        self.tool_executor = ToolExecutor(
            self.tool_registry,
            config=self.context.config.tool,
        )

        # Wire event bus
        self.set_event_bus(self.context.event_bus)
        self.tool_executor.set_event_bus(self.context.event_bus)
        self.llm.set_event_bus(self.context.event_bus)

    @property
    def tools(self) -> list[BaseTool]:
        return self.tool_registry.list_tools()

    @property
    def tool_schemas(self) -> list[ToolSchema]:
        return self.tool_registry.get_schemas()

    async def run(self, input: str | Message) -> AgentResult:
        """Main execution loop.

        Repeatedly calls step() until:
        1. step() returns a final response, or
        2. max_steps is reached.

        Safe to call multiple times — state is reset automatically when
        the agent is in a terminal state (FINISHED or ERROR).
        """
        # Reset state for agent reuse (e.g., team orchestration, pipelines)
        if self.context.state.is_terminal:
            self.context.state.reset()

        # Normalize input
        if isinstance(input, str):
            input_msg = Message.user(input)
            input_text = input
        else:
            input_msg = input
            input_text = input.content or ""

        # Initialize context
        if self.system_prompt:
            await self.context.short_term_memory.add_message(
                Message.system(self.system_prompt)
            )
        await self.context.short_term_memory.add_message(input_msg)
        self.context.state.transition(AgentState.THINKING)

        await self.hooks.on_run_start(self.name, input_text)
        await self.emit("agent.run.start", agent=self.name, input=input_text)

        steps: list[StepResult] = []
        self._total_usage = Usage()
        final_output = ""

        try:
            for step_num in range(1, self.max_steps + 1):
                await self.hooks.on_step_start(self.name, step_num)
                await self.emit("agent.step.start", agent=self.name, step=step_num)

                step_result = await self.step()
                steps.append(step_result)

                await self.hooks.on_step_end(self.name, step_num)
                await self.emit("agent.step.end", agent=self.name, step=step_num)

                if step_result.response is not None:
                    final_output = step_result.response
                    self.context.state.transition(AgentState.FINISHED)
                    break
            else:
                # max_steps exceeded
                self.context.state.transition(AgentState.ERROR)
                raise MaxStepsExceededError(
                    f"Agent '{self.name}' exceeded {self.max_steps} steps"
                )

        except Exception as e:
            await self.hooks.on_error(self.name, e)
            await self.emit("agent.run.error", agent=self.name, error=str(e))
            if not isinstance(e, MaxStepsExceededError):
                if not self.context.state.is_terminal:
                    self.context.state.transition(AgentState.ERROR)
            raise

        messages = await self.context.short_term_memory.get_context_messages()
        result = AgentResult(
            output=final_output,
            messages=messages,
            steps=steps,
            usage=self._total_usage,
        )

        await self.hooks.on_run_end(self.name, final_output)
        await self.emit("agent.run.end", agent=self.name, output=final_output, steps=len(steps))
        return result

    @abstractmethod
    async def step(self) -> StepResult:
        """Execute a single reasoning step.

        Subclasses implement their strategy here:
        - ReActAgent: think -> act -> observe
        - PlanAgent: plan -> execute step
        - ConversationalAgent: generate response

        Returns:
            StepResult. If response is not None, the run loop ends.
        """
        ...

    async def call_llm(
        self,
        messages: list[Message] | None = None,
        tools: list[ToolSchema] | None = None,
        use_long_term: bool = False,
        long_term_query: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call the LLM with current context messages or provided messages.

        Args:
            messages: Override messages. If None, uses short-term memory.
            tools: Override tool schemas. If None, uses registered tools.
            use_long_term: Query long-term memory and inject results.
            long_term_query: Custom query for long-term retrieval.
            **kwargs: Passed through to llm.generate_with_events().
        """
        messages = await self.context.build_llm_messages(
            base_messages=messages,
            include_working=True,
            include_long_term=use_long_term,
            long_term_query=long_term_query,
        )

        if tools is None and self.tool_schemas:
            tools = self.tool_schemas

        await self.hooks.on_llm_call(self.name, messages)
        if self.context.state.current != AgentState.THINKING:
            self.context.state.transition(AgentState.THINKING)

        response = await self.llm.generate_with_events(messages, tools=tools, **kwargs)

        # Accumulate token usage
        if hasattr(self, '_total_usage'):
            self._total_usage = self._total_usage + response.usage

        # Store assistant message in memory
        await self.context.short_term_memory.add_message(response.message)
        return response

    async def execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls and store results in memory."""
        self.context.state.transition(AgentState.ACTING)

        for tc in tool_calls:
            await self.hooks.on_tool_call(self.name, tc)

        results = await self.tool_executor.execute_batch(tool_calls)

        self.context.state.transition(AgentState.OBSERVING)

        for result in results:
            await self.hooks.on_tool_result(self.name, result)
            # Store tool result as a message
            await self.context.short_term_memory.add_message(
                Message.tool(
                    tool_call_id=result.tool_call_id,
                    content=result.content,
                    is_error=result.is_error,
                )
            )

        return results

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} tools={len(self.tools)}>"
