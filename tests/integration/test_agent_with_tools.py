"""Integration tests: agent + tools working together."""
from __future__ import annotations

import pytest

from agent_harness.agent.react import ReActAgent
from agent_harness.context.context import AgentContext
from agent_harness.tool import tool


@tool
def calculator(a: int, b: int) -> str:
    """Add two numbers.

    Args:
        a: First number
        b: Second number
    """
    return str(a + b)


@tool(name="multiplier", description="Multiply two numbers")
def multiply(x: int, y: int) -> str:
    """Multiply two numbers.

    Args:
        x: First factor
        y: Second factor
    """
    return str(x * y)


@pytest.mark.asyncio
async def test_react_agent_executes_tool_and_returns_result(mock_llm, config):
    """ReActAgent should call the calculator tool, then return a final answer."""
    mock_llm.add_tool_call_response("calculator", {"a": 3, "b": 4})
    mock_llm.add_text_response("The answer is 7.")

    ctx = AgentContext.create(config)
    agent = ReActAgent(name="calc_agent", llm=mock_llm, tools=[calculator], context=ctx)

    result = await agent.run("What is 3 + 4?")

    assert result.output == "The answer is 7."
    assert result.step_count == 2
    # First step: tool call; second step: final answer
    assert result.steps[0].action is not None
    assert result.steps[0].observation is not None
    assert result.steps[0].observation[0].content == "7"
    assert result.steps[1].response == "The answer is 7."
    # LLM was called twice
    assert len(mock_llm.call_history) == 2


@pytest.mark.asyncio
async def test_react_agent_with_multiple_tools(mock_llm, config):
    """ReActAgent should pick the right tool when multiple are registered."""
    mock_llm.add_tool_call_response("multiplier", {"x": 5, "y": 6})
    mock_llm.add_text_response("5 times 6 is 30.")

    ctx = AgentContext.create(config)
    agent = ReActAgent(
        name="multi_tool_agent",
        llm=mock_llm,
        tools=[calculator, multiply],
        context=ctx,
    )

    result = await agent.run("What is 5 * 6?")

    assert result.output == "5 times 6 is 30."
    assert result.steps[0].observation[0].content == "30"


@pytest.mark.asyncio
async def test_react_agent_direct_answer_without_tools(mock_llm, config):
    """ReActAgent should return immediately when LLM doesn't invoke any tool."""
    mock_llm.add_text_response("Hello! How can I help you?")

    ctx = AgentContext.create(config)
    agent = ReActAgent(name="greeter", llm=mock_llm, tools=[calculator], context=ctx)

    result = await agent.run("Hi there")

    assert result.output == "Hello! How can I help you?"
    assert result.step_count == 1
    assert result.steps[0].action is None
