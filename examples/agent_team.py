"""AgentTeam with three collaboration modes: supervisor, debate, round_robin.

Demonstrates multi-agent collaboration where specialist agents contribute
different perspectives and are coordinated via different strategies.
"""

import asyncio
import os

from agent_harness import ConversationalAgent, HarnessConfig
from agent_harness.llm import OpenAIProvider
from agent_harness.core import LLMConfig
from agent_harness.orchestration import AgentTeam, TeamMode


def create_agents(llm: OpenAIProvider) -> tuple:
    """Create the specialist agents and supervisor."""
    optimist = ConversationalAgent(
        name="optimist",
        llm=llm,
        system_prompt=(
            "You are an optimistic analyst. You always highlight opportunities, "
            "potential upsides, and reasons for enthusiasm. Keep it to 2-3 sentences."
        ),
    )

    pessimist = ConversationalAgent(
        name="pessimist",
        llm=llm,
        system_prompt=(
            "You are a cautious, risk-focused analyst. You highlight potential "
            "downsides, risks, and pitfalls. Keep it to 2-3 sentences."
        ),
    )

    realist = ConversationalAgent(
        name="realist",
        llm=llm,
        system_prompt=(
            "You are a pragmatic realist. You weigh both sides objectively and "
            "provide a balanced, evidence-based view. Keep it to 2-3 sentences."
        ),
    )

    supervisor = ConversationalAgent(
        name="supervisor",
        llm=llm,
        system_prompt=(
            "You are a senior decision-maker. Given input from multiple analysts, "
            "synthesize their perspectives into a clear, actionable recommendation. "
            "Be concise — one short paragraph."
        ),
    )

    return optimist, pessimist, realist, supervisor


async def demo_supervisor(agents: tuple, topic: str) -> None:
    """Supervisor mode: delegates to all workers, then synthesizes."""
    optimist, pessimist, realist, supervisor = agents

    print("=" * 60)
    print("MODE 1: Supervisor")
    print("=" * 60)

    team = AgentTeam(
        agents=[optimist, pessimist, realist],
        mode=TeamMode.SUPERVISOR,
        supervisor=supervisor,
    )

    result = await team.run(topic)
    print(f"\nSynthesized output:\n{result.output}")
    print(f"Rounds: {result.rounds} | Contributors: {list(result.agent_results.keys())}")


async def demo_debate(agents: tuple, topic: str) -> None:
    """Debate mode: agents argue independently, judge picks best."""
    optimist, pessimist, realist, supervisor = agents

    print("\n" + "=" * 60)
    print("MODE 2: Debate")
    print("=" * 60)

    team = AgentTeam(
        agents=[optimist, pessimist, realist],
        mode=TeamMode.DEBATE,
        supervisor=supervisor,
    )

    result = await team.run(topic)
    print(f"\nJudge's verdict:\n{result.output}")
    print(f"Rounds: {result.rounds} | Debaters: {list(result.agent_results.keys())}")


async def demo_round_robin(agents: tuple, topic: str) -> None:
    """Round-robin mode: agents build on each other's work iteratively."""
    optimist, pessimist, realist, _ = agents

    print("\n" + "=" * 60)
    print("MODE 3: Round Robin")
    print("=" * 60)

    team = AgentTeam(
        agents=[optimist, pessimist, realist],
        mode=TeamMode.ROUND_ROBIN,
        max_rounds=2,
    )

    result = await team.run(topic)
    print(f"\nFinal output (after iterative refinement):\n{result.output}")
    print(f"Rounds: {result.rounds} | Participants: {list(result.agent_results.keys())}")


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Set OPENAI_API_KEY environment variable to run this demo.")
        return

    llm = OpenAIProvider(LLMConfig(provider="openai", model="gpt-4o"))
    agents = create_agents(llm)

    topic = "Should a mid-size company invest heavily in AI automation this year?"

    print(f"Topic: {topic}\n")

    await demo_supervisor(agents, topic)
    await demo_debate(agents, topic)
    await demo_round_robin(agents, topic)


if __name__ == "__main__":
    asyncio.run(main())
