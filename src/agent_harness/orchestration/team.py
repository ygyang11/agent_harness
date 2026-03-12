"""Multi-agent team collaboration patterns."""
from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.agent.base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class TeamMode(str, Enum):
    """Team collaboration modes."""
    SUPERVISOR = "supervisor"
    DEBATE = "debate"
    ROUND_ROBIN = "round_robin"


class TeamResult(BaseModel):
    """Result from team execution."""
    output: str
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    rounds: int = 0


class AgentTeam:
    """Multi-agent team with different collaboration modes.

    Modes:
        supervisor: A supervisor agent delegates subtasks and synthesizes results.
        debate: Multi-round debate, agents refine positions seeing others' views, judge synthesizes.
        round_robin: Agents take turns; early termination when all respond DONE.

    Example:
        team = AgentTeam(
            agents=[researcher, analyst, writer],
            mode=TeamMode.SUPERVISOR,
            supervisor=manager_agent,
        )
        result = await team.run("Analyze the impact of AI on healthcare")
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        mode: str | TeamMode = TeamMode.SUPERVISOR,
        supervisor: BaseAgent | None = None,
        max_rounds: int = 3,
    ) -> None:
        self.agents = agents
        self.mode = TeamMode(mode) if isinstance(mode, str) else mode
        self.supervisor = supervisor
        self.max_rounds = max_rounds

        if self.mode == TeamMode.SUPERVISOR and not supervisor:
            raise ValueError("Supervisor mode requires a supervisor agent")

    async def run(self, input: str, hooks: Any = None) -> TeamResult:
        if hasattr(hooks, "on_team_start"):
            await hooks.on_team_start("team", self.mode.value)

        if self.mode == TeamMode.SUPERVISOR:
            result = await self._run_supervisor(input)
        elif self.mode == TeamMode.DEBATE:
            result = await self._run_debate(input)
        elif self.mode == TeamMode.ROUND_ROBIN:
            result = await self._run_round_robin(input)
        else:
            raise ValueError(f"Unknown team mode: {self.mode}")

        if hasattr(hooks, "on_team_end"):
            await hooks.on_team_end("team", self.mode.value)

        return result

    async def _run_supervisor(self, input: str) -> TeamResult:
        """Supervisor delegates subtasks to workers, then synthesizes results."""
        from agent_harness.core.message import Message
        from agent_harness.utils.json_utils import parse_json_lenient

        assert self.supervisor is not None
        agent_results: dict[str, AgentResult] = {}

        # Step 1: Supervisor analyses task and assigns subtasks
        worker_info = "\n".join(
            f"- {a.name}: {a.system_prompt[:100]}..." if a.system_prompt else f"- {a.name}"
            for a in self.agents
        )

        delegation_input = (
            f"You are a supervisor managing a team of agents.\n"
            f"Available workers:\n{worker_info}\n\n"
            f"Task: {input}\n\n"
            f"Analyze the task and assign subtasks to workers. "
            f"Respond with a JSON object mapping worker names to their subtasks:\n"
            f'{{"assignments": {{"worker_name": "subtask description", ...}}}}'
        )

        delegation_result = await self.supervisor.run(delegation_input)
        agent_results["supervisor_delegation"] = delegation_result

        # Step 2: Parse assignments and run workers in parallel
        assignments: dict[str, str] | None = None
        try:
            data = parse_json_lenient(delegation_result.output)
            if isinstance(data, dict):
                assignments = data.get("assignments", data)
                if not isinstance(assignments, dict):
                    assignments = None
        except (ValueError, TypeError):
            pass

        worker_map = {a.name: a for a in self.agents}

        if assignments:
            # Run assigned subtasks in parallel
            tasks = []
            task_names = []
            for worker_name, subtask in assignments.items():
                agent = worker_map.get(worker_name)
                if agent:
                    tasks.append(agent.run(str(subtask)))
                    task_names.append(worker_name)

            if not tasks:
                # No valid worker names; fallback to all workers with original input
                tasks = [a.run(input) for a in self.agents]
                task_names = [a.name for a in self.agents]
        else:
            # JSON parse failed; fallback to all workers with original input
            logger.warning("Supervisor delegation JSON parse failed, running all workers")
            tasks = [a.run(input) for a in self.agents]
            task_names = [a.name for a in self.agents]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.warning("Worker '%s' failed: %s", name, result)
                agent_results[name] = AgentResult(
                    output=f"Error: {result}", messages=[], steps=[],
                )
            else:
                agent_results[name] = result

        # Step 3: Supervisor synthesizes results
        synthesis_input = f"Original task: {input}\n\nWorker results:\n"
        for name in task_names:
            synthesis_input += f"\n--- {name} ---\n{agent_results[name].output}\n"
        synthesis_input += "\nSynthesize these results into a comprehensive final answer."

        final = await self.supervisor.run(synthesis_input)
        agent_results["supervisor_synthesis"] = final

        return TeamResult(
            output=final.output,
            agent_results=agent_results,
            rounds=1,
        )

    async def _run_debate(self, input: str) -> TeamResult:
        """Multi-round debate: independent first round, then refine seeing others' views."""
        agent_results: dict[str, AgentResult] = {}

        # Round 1: Independent answers in parallel
        tasks = [agent.run(input) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        current_positions: dict[str, str] = {}
        for agent, result in zip(self.agents, results):
            if isinstance(result, Exception):
                logger.warning("Agent '%s' failed in debate round 1: %s", agent.name, result)
                current_positions[agent.name] = f"Error: {result}"
                agent_results[f"{agent.name}_round1"] = AgentResult(
                    output=f"Error: {result}", messages=[], steps=[],
                )
            else:
                current_positions[agent.name] = result.output
                agent_results[f"{agent.name}_round1"] = result

        # Rounds 2..max_rounds: Each agent sees others' positions and refines
        for round_num in range(2, self.max_rounds + 1):
            round_tasks = []
            for agent in self.agents:
                other_views = "\n".join(
                    f"--- {name} ---\n{pos}"
                    for name, pos in current_positions.items()
                    if name != agent.name
                )
                refine_input = (
                    f"Original question: {input}\n\n"
                    f"Your previous position:\n{current_positions.get(agent.name, '')}\n\n"
                    f"Other agents' positions:\n{other_views}\n\n"
                    f"This is debate round {round_num}/{self.max_rounds}. "
                    f"Consider other viewpoints and refine your position. "
                    f"You may change your mind or strengthen your argument."
                )
                round_tasks.append(agent.run(refine_input))

            round_results = await asyncio.gather(*round_tasks, return_exceptions=True)
            for agent, result in zip(self.agents, round_results):
                key = f"{agent.name}_round{round_num}"
                if isinstance(result, Exception):
                    logger.warning("Agent '%s' failed in debate round %d: %s", agent.name, round_num, result)
                    agent_results[key] = AgentResult(
                        output=f"Error: {result}", messages=[], steps=[],
                    )
                else:
                    current_positions[agent.name] = result.output
                    agent_results[key] = result

        # Final: Judge synthesizes
        judge = self.supervisor or self.agents[-1]
        judge_input = f"Original question: {input}\n\nFinal positions after {self.max_rounds} rounds of debate:\n"
        for name, pos in current_positions.items():
            judge_input += f"\n--- {name} ---\n{pos}\n"
        judge_input += (
            "\nEvaluate these final positions. Synthesize the best parts into a comprehensive answer."
        )

        final = await judge.run(judge_input)
        agent_results["judge"] = final

        return TeamResult(
            output=final.output,
            agent_results=agent_results,
            rounds=self.max_rounds,
        )

    async def _run_round_robin(self, input: str) -> TeamResult:
        """Agents take turns with early termination when all respond DONE."""
        agent_results: dict[str, AgentResult] = {}
        accumulated = f"Task: {input}\n"
        actual_rounds = 0

        for round_num in range(1, self.max_rounds + 1):
            actual_rounds = round_num
            all_done = True

            for agent in self.agents:
                agent_input = (
                    f"{accumulated}\n"
                    f"You are '{agent.name}'. Provide your contribution. "
                    f"Build on previous agents' work if available.\n"
                    f"If the task has been sufficiently completed, respond with exactly 'DONE'."
                )
                result = await agent.run(agent_input)
                key = f"{agent.name}_round{round_num}"
                agent_results[key] = result
                accumulated += f"\n--- {agent.name} (round {round_num}) ---\n{result.output}\n"

                if result.output.strip().upper() != "DONE":
                    all_done = False

            if all_done:
                logger.info("Round robin: all agents responded DONE at round %d", round_num)
                break

        # Final output is the last non-DONE agent's output, or the last output
        last_output = ""
        for key in reversed(list(agent_results.keys())):
            if agent_results[key].output.strip().upper() != "DONE":
                last_output = agent_results[key].output
                break
        if not last_output:
            last_output = list(agent_results.values())[-1].output

        return TeamResult(
            output=last_output,
            agent_results=agent_results,
            rounds=actual_rounds,
        )
