"""PlanAgent for complex task decomposition and multi-step execution.

Demonstrates: PlanAgent's three-phase workflow — planning a JSON plan,
executing each step with tool access, and synthesizing a final answer.
"""

import asyncio
import os

from agent_harness import PlanAgent, tool, HarnessConfig
from agent_harness.llm import OpenAIProvider
from agent_harness.core import LLMConfig


@tool
async def search_topic(query: str) -> str:
    """Search for information on a topic.

    Args:
        query: The search query string.
    """
    # Simulated search results
    results = {
        "renewable energy trends 2024": (
            "Solar capacity grew 30% globally. Wind power investment reached $120B. "
            "Battery storage costs dropped 15%. Green hydrogen projects doubled."
        ),
        "solar energy statistics": (
            "Global solar capacity: 1.6 TW. Top producers: China, US, India. "
            "Average cost: $0.05/kWh. Expected to triple by 2030."
        ),
        "wind energy statistics": (
            "Global wind capacity: 900 GW. Offshore wind growing at 25%/year. "
            "Top markets: China, Europe, US. Jobs: 1.4 million worldwide."
        ),
    }
    for key, value in results.items():
        if any(word in query.lower() for word in key.split()):
            return value
    return f"Search results for '{query}': General information available on this topic."


@tool
async def analyze_data(data: str) -> str:
    """Analyze provided data and extract key insights.

    Args:
        data: The raw data or text to analyze.
    """
    word_count = len(data.split())
    return (
        f"Analysis of {word_count}-word input: "
        f"Identified 3 key trends, 2 statistical claims, and 1 projection. "
        f"Data appears reliable with multiple corroborating sources."
    )


@tool
async def write_summary(topic: str, key_points: str) -> str:
    """Write a structured summary given a topic and key points.

    Args:
        topic: The topic to summarize.
        key_points: Comma-separated key points to include.
    """
    return (
        f"Summary: {topic}\n"
        f"Key findings: {key_points}\n"
        f"Conclusion: The data supports continued growth in this sector "
        f"with strong investment signals and declining costs."
    )


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Set OPENAI_API_KEY environment variable to run this demo.")
        return

    llm = OpenAIProvider(LLMConfig(provider="openai", model="gpt-4o"))

    agent = PlanAgent(
        name="researcher",
        llm=llm,
        tools=[search_topic, analyze_data, write_summary],
        allow_replan=True,
    )

    query = (
        "Research the current state of renewable energy and produce "
        "a brief report covering solar, wind, and overall trends."
    )
    print(f"Query: {query}\n")

    result = await agent.run(query)

    print(f"Final Report:\n{result.output}\n")
    print(f"Total steps: {result.step_count}")
    print(f"Tokens used: {result.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
