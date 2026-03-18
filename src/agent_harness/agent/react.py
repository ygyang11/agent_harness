"""ReAct agent: Reasoning + Acting loop."""
from __future__ import annotations

import logging

from agent_harness.agent.base import BaseAgent, StepResult
from agent_harness.llm.types import FinishReason

logger = logging.getLogger(__name__)

REACT_PROMPTS: dict[str, str] = {
    "system.default": """You are a helpful AI assistant with access to tools.

When you need information or need to take action, use the available tools.
Think step by step about what you need to do, then take action.
When you have enough information to answer the user's question, provide a final response.

Important:
- Use tools when you need external information or capabilities
- You can call multiple tools at once if they are independent
- After receiving tool results, analyze them before deciding next steps
- Provide a clear, comprehensive final answer when ready""",
}


class ReActAgent(BaseAgent):
    """ReAct agent implementing the Reasoning + Acting paradigm.

    Execution loop:
    1. THINK: LLM reasons about the current state and decides what to d
    2. ACT: If LLM calls tools, execute them
    3. OBSERVE: Feed tool results back to LLM
    4. Repeat until LLM provides a final answer (no tool calls)

    Supports:
    - Parallel tool calls (when LLM returns multiple tool_calls)
    - Automatic thought chain tracking
    - Configurable system prompt
    - Force tool usage via tool_choice

    Example:
        agent = ReActAgent(
            name="researcher",
            llm=openai_provider,
            tools=[search_tool, calculator_tool],
        )
        result = await agent.run("What is the population of France?")
    """

    def __init__(self, system_prompt: str | None = None, **kwargs):
        if system_prompt is None:
            system_prompt = REACT_PROMPTS["system.default"]
        super().__init__(system_prompt=system_prompt, **kwargs)

    async def step(self) -> StepResult:
        """Execute one ReAct cycle: Think -> (Act -> Observe)? -> Response?"""
        # THINK: Call LLM with current context and available tools
        response = await self.call_llm()

        # Check if LLM wants to call tools
        if response.has_tool_calls:
            tool_calls = response.message.tool_calls or []

            logger.debug(
                "Agent '%s' calling %d tool(s): %s",
                self.name,
                len(tool_calls),
                [tc.name for tc in tool_calls],
            )

            # ACT: Execute tools
            results = await self.execute_tools(tool_calls)

            # OBSERVE: Results are now in short-term memory
            # Return step without response — loop continues
            return StepResult(
                thought=response.message.content,
                action=tool_calls,
                observation=results,
            )

        # No tool calls — LLM is providing a final answer
        return StepResult(
            thought=None,
            response=response.message.content or "",
        )
