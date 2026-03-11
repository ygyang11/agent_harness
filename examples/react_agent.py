"""ReAct Agent with tool calling — the most fundamental agent pattern.

Demonstrates: @tool decorator, OpenAIProvider, ReActAgent, running a query
that requires multi-step tool use, and inspecting execution steps.
"""

import asyncio
import os
import random

from agent_harness import ReActAgent, tool, HarnessConfig
from agent_harness.llm import OpenAIProvider
from agent_harness.core import LLMConfig


@tool
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A Python math expression to evaluate, e.g. '2 + 3 * 4'.
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city to get weather for.
    """
    # Simulated weather data
    conditions = ["sunny", "partly cloudy", "overcast", "rainy"]
    temp = random.randint(15, 35)
    return f"{city}: {temp}°C, {random.choice(conditions)}"


@tool
async def get_population(country: str) -> str:
    """Look up the approximate population of a country.

    Args:
        country: Name of the country.
    """
    populations = {
        "france": "68 million",
        "germany": "84 million",
        "japan": "125 million",
        "brazil": "214 million",
        "india": "1.4 billion",
    }
    return populations.get(country.lower(), f"Population data not available for {country}")


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Set OPENAI_API_KEY environment variable to run this demo.")
        return

    llm = OpenAIProvider(LLMConfig(provider="openai", model="gpt-4o"))

    agent = ReActAgent(
        name="assistant",
        llm=llm,
        tools=[calculate, get_weather, get_population],
    )

    query = (
        "What's the weather in Paris and Tokyo? "
        "Also, what is the population of France divided by 4?"
    )
    print(f"Query: {query}\n")

    result = await agent.run(query)

    print(f"Answer:\n{result.output}\n")
    print(f"Steps taken: {result.step_count}")
    print(f"Tokens used: {result.usage.total_tokens}")

    for i, step in enumerate(result.steps, 1):
        print(f"\n--- Step {i} ---")
        if step.action:
            for tc in step.action:
                print(f"  Tool: {tc.name}({tc.arguments})")
        if step.observation:
            for obs in step.observation:
                print(f"  Result: {obs.content}")
        if step.response:
            print(f"  Response: {step.response[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
