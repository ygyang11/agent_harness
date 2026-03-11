"""Deep Research — the framework's headline use case.

Architecture:
  1. PlanAgent creates a research plan from the user query.
  2. Multiple ReActAgents execute search/analysis tasks in parallel (DAG).
  3. ConversationalAgent synthesizes the final report.

This demo wires together PlanAgent, ReActAgent, ConversationalAgent,
DAG orchestration, and Pipeline into a complete research workflow.
"""

import asyncio
import os
import random

from agent_harness import ReActAgent, PlanAgent, ConversationalAgent, tool, HarnessConfig
from agent_harness.llm import OpenAIProvider
from agent_harness.core import LLMConfig
from agent_harness.orchestration import DAGOrchestrator, DAGNode, Pipeline, PipelineStep
from agent_harness.context import AgentContext


# ---------------------------------------------------------------------------
# Tools available to research agents
# ---------------------------------------------------------------------------

@tool
async def web_search(query: str, max_results: int = 3) -> str:
    """Search the web and return summarized results.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.
    """
    snippets = {
        "quantum computing breakthroughs": [
            "IBM unveils 1,121-qubit Condor processor, marking a milestone in quantum hardware.",
            "Google claims quantum supremacy with new error-correction techniques.",
            "Startup QuEra demonstrates neutral-atom quantum computer with 48 logical qubits.",
        ],
        "quantum computing applications": [
            "Drug discovery: quantum simulations model molecular interactions 100x faster.",
            "Finance: portfolio optimization using quantum annealing shows 30% improvement.",
            "Cryptography: NIST finalizes post-quantum encryption standards.",
        ],
        "quantum computing challenges": [
            "Decoherence remains the primary obstacle — qubits lose state in microseconds.",
            "Error rates above 0.1% make practical computation infeasible for most tasks.",
            "Talent shortage: fewer than 5,000 quantum engineers worldwide.",
        ],
        "quantum computing market": [
            "Global quantum computing market projected to reach $65B by 2030.",
            "Major players: IBM, Google, Microsoft, IonQ, Rigetti, D-Wave.",
            "Government investment: US $1.2B, EU €1B, China $15B committed.",
        ],
    }
    for key, values in snippets.items():
        if any(word in query.lower() for word in key.split()):
            selected = values[:max_results]
            return "\n".join(f"- {s}" for s in selected)
    return f"Found {max_results} results for '{query}': General information available."


@tool
async def read_article(url: str) -> str:
    """Read and extract content from a web article.

    Args:
        url: URL of the article to read.
    """
    # Simulated article content
    return (
        f"Article from {url}: This is a comprehensive piece covering recent "
        f"developments, expert opinions, and statistical data. Key takeaway: "
        f"the field is advancing rapidly with {random.randint(3, 8)} major "
        f"milestones achieved in the past quarter."
    )


@tool
async def take_notes(topic: str, content: str) -> str:
    """Save research notes on a specific topic.

    Args:
        topic: The topic these notes relate to.
        content: The notes to save.
    """
    word_count = len(content.split())
    return f"Saved {word_count}-word note on '{topic}'. Notes are available for synthesis."


# ---------------------------------------------------------------------------
# Build the deep research pipeline
# ---------------------------------------------------------------------------

async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Set OPENAI_API_KEY environment variable to run this demo.")
        return

    llm = OpenAIProvider(LLMConfig(provider="openai", model="gpt-4o"))
    research_tools = [web_search, read_article, take_notes]
    research_query = "What is the current state of quantum computing and its potential impact?"

    print(f"Research query: {research_query}\n")
    print("Phase 1: Planning research...")

    # --- Phase 1: Plan the research ---
    planner = PlanAgent(
        name="planner",
        llm=llm,
        tools=research_tools,
        system_prompt=(
            "You are a research planner. Given a research question, create a "
            "plan with 3-4 concrete steps. Use your tools to search for initial "
            "information, then structure a research plan."
        ),
        max_steps=15,
    )
    plan_result = await planner.run(research_query)
    print(f"Plan complete ({plan_result.step_count} steps)\n")

    # --- Phase 2: Parallel deep-dive research via DAG ---
    print("Phase 2: Parallel research execution...")

    hardware_researcher = ReActAgent(
        name="hardware_researcher",
        llm=llm,
        tools=research_tools,
        system_prompt=(
            "You research quantum computing hardware and breakthroughs. "
            "Search for recent developments, read articles, and take notes. "
            "Provide a concise summary of your findings."
        ),
    )

    applications_researcher = ReActAgent(
        name="applications_researcher",
        llm=llm,
        tools=research_tools,
        system_prompt=(
            "You research practical applications of quantum computing. "
            "Search for use cases in industry, read articles, and take notes. "
            "Provide a concise summary of your findings."
        ),
    )

    market_researcher = ReActAgent(
        name="market_researcher",
        llm=llm,
        tools=research_tools,
        system_prompt=(
            "You research the quantum computing market and investment landscape. "
            "Search for market data, read articles, and take notes. "
            "Provide a concise summary of your findings."
        ),
    )

    def merge_research(results: dict) -> str:
        sections = []
        for node_id in ["hardware", "applications", "market"]:
            if node_id in results:
                sections.append(f"## {node_id.title()} Research\n{results[node_id].output}")
        return (
            "Synthesize the following research into a comprehensive, well-structured "
            "report with an introduction, key findings, and conclusion:\n\n"
            + "\n\n".join(sections)
        )

    # The synthesizer combines all research branches
    synthesizer = ConversationalAgent(
        name="report_writer",
        llm=llm,
        system_prompt=(
            "You are an expert technical writer. Given research from multiple "
            "analysts, write a polished, well-structured research report. "
            "Include an executive summary, key findings organized by theme, "
            "and a forward-looking conclusion."
        ),
    )

    dag = DAGOrchestrator(
        nodes=[
            DAGNode(id="hardware", agent=hardware_researcher),
            DAGNode(id="applications", agent=applications_researcher),
            DAGNode(id="market", agent=market_researcher),
            DAGNode(
                id="report",
                agent=synthesizer,
                dependencies=["hardware", "applications", "market"],
                input_transform=merge_research,
            ),
        ]
    )

    dag_result = await dag.run(research_query)

    print(f"Execution order: {dag_result.execution_order}")
    for node_id, node_result in dag_result.outputs.items():
        if node_id != "report":
            print(f"  [{node_id}]: {node_result.step_count} steps, "
                  f"{node_result.usage.total_tokens} tokens")

    # --- Phase 3: Final report ---
    print("\n" + "=" * 60)
    print("FINAL RESEARCH REPORT")
    print("=" * 60)
    print(dag_result.outputs["report"].output)

    # Summary stats
    total_tokens = sum(r.usage.total_tokens for r in dag_result.outputs.values())
    total_tokens += plan_result.usage.total_tokens
    print(f"\n--- Stats ---")
    print(f"Planning tokens: {plan_result.usage.total_tokens}")
    print(f"Research + synthesis tokens: {sum(r.usage.total_tokens for r in dag_result.outputs.values())}")
    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
