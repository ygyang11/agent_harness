"""Multi-agent team collaboration patterns."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.agent.base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class TeamResult(BaseModel):
    """Result from team execution."""
    output: str
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    rounds: int = 0


class AgentTeam:
    """Multi-agent team with different collaboration modes.

    Modes:
        supervisor: A supervisor agent delegates to and synthesizes from workers.
        debate: Agents independently answer, then a judge picks or synthesizes.
        round_robin: Agents take turns, each seeing previous outputs.

    Example:
        team = AgentTeam(
            agents=[researcher, analyst, writer],
            mode="supervisor",
            supervisor=manager_agent,
        )
        result = await team.run("Analyze the impact of AI on healthcare")
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        mode: str = "supervisor",
        supervisor: BaseAgent | None = None,
        max_rounds: int = 3,
    ) -> None:
        self.agents = agents
        self.mode = mode
        self.supervisor = supervisor
        self.max_rounds = max_rounds

        if mode == "supervisor" and not supervisor:
            raise ValueError("Supervisor mode requires a supervisor agent")

    async def run(self, input: str) -> TeamResult:
        if self.mode == "supervisor":
            return await self._run_supervisor(input)
        elif self.mode == "debate":
            return await self._run_debate(input)
        elif self.mode == "round_robin":
            return await self._run_round_robin(input)
        else:
            raise ValueError(f"Unknown team mode: {self.mode}")

    async def _run_supervisor(self, input: str) -> TeamResult:
        """Supervisor delegates tasks and synthesizes results."""
        import asyncio

        assert self.supervisor is not None
        agent_results: dict[str, AgentResult] = {}

        # Build worker descriptions for supervisor
        worker_info = "\n".join(
            f"- {a.name}: {a.system_prompt[:100]}..." if a.system_prompt else f"- {a.name}"
            for a in self.agents
        )

        supervisor_input = (
            f"You are a supervisor managing a team of agents.\n"
            f"Available workers:\n{worker_info}\n\n"
            f"Task: {input}\n\n"
            f"Delegate subtasks to workers, then synthesize their results into a final answer."
        )

        # Run all workers in parallel with the original task
        tasks = [agent.run(input) for agent in self.agents]
        results = await asyncio.gather(*tasks)

        for agent, result in zip(self.agents, results):
            agent_results[agent.name] = result

        # Synthesize with supervisor
        synthesis_input = f"Original task: {input}\n\nWorker results:\n"
        for name, result in agent_results.items():
            synthesis_input += f"\n--- {name} ---\n{result.output}\n"
        synthesis_input += "\nSynthesize these results into a comprehensive final answer."

        final = await self.supervisor.run(synthesis_input)
        agent_results["supervisor"] = final

        return TeamResult(
            output=final.output,
            agent_results=agent_results,
            rounds=1,
        )

    async def _run_debate(self, input: str) -> TeamResult:
        """Each agent answers independently, last agent (as judge) synthesizes."""
        import asyncio

        agent_results: dict[str, AgentResult] = {}

        # All agents answer independently in parallel
        tasks = [agent.run(input) for agent in self.agents]
        results = await asyncio.gather(*tasks)

        for agent, result in zip(self.agents, results):
            agent_results[agent.name] = result

        # Use the last agent as judge if no supervisor
        judge = self.supervisor or self.agents[-1]

        judge_input = f"Original question: {input}\n\nMultiple answers were provided:\n"
        for name, result in agent_results.items():
            judge_input += f"\n--- {name} ---\n{result.output}\n"
        judge_input += (
            "\nEvaluate these answers. Synthesize the best parts into a final, comprehensive answer."
        )

        final = await judge.run(judge_input)
        agent_results["judge"] = final

        return TeamResult(
            output=final.output,
            agent_results=agent_results,
            rounds=1,
        )

    async def _run_round_robin(self, input: str) -> TeamResult:
        """Agents take turns, each seeing all previous outputs."""
        agent_results: dict[str, AgentResult] = {}
        accumulated = f"Task: {input}\n"

        for round_num in range(1, self.max_rounds + 1):
            for agent in self.agents:
                agent_input = (
                    f"{accumulated}\n"
                    f"You are '{agent.name}'. Provide your contribution. "
                    f"Build on previous agents' work if available."
                )
                result = await agent.run(agent_input)
                key = f"{agent.name}_round{round_num}"
                agent_results[key] = result
                accumulated += f"\n--- {agent.name} (round {round_num}) ---\n{result.output}\n"

        # Final output is the last agent's last output
        last_key = list(agent_results.keys())[-1]
        return TeamResult(
            output=agent_results[last_key].output,
            agent_results=agent_results,
            rounds=self.max_rounds,
        )
