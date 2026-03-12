"""Tests for orchestration: TeamMode enum and Router with LLM."""
from __future__ import annotations

import pytest

from agent_harness.orchestration.team import AgentTeam, TeamMode, TeamResult


class TestTeamMode:
    def test_enum_values(self) -> None:
        assert TeamMode.SUPERVISOR.value == "supervisor"
        assert TeamMode.DEBATE.value == "debate"
        assert TeamMode.ROUND_ROBIN.value == "round_robin"

    def test_string_construction(self) -> None:
        assert TeamMode("supervisor") == TeamMode.SUPERVISOR
        assert TeamMode("debate") == TeamMode.DEBATE
        assert TeamMode("round_robin") == TeamMode.ROUND_ROBIN

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError):
            TeamMode("invalid_mode")

    def test_team_init_accepts_string(self) -> None:
        """AgentTeam should accept string or enum mode values."""
        from tests.conftest import MockLLM
        from agent_harness.agent.conversational import ConversationalAgent

        llm = MockLLM()
        llm.add_text_response("result")
        supervisor = ConversationalAgent(name="sup", llm=llm)
        worker = ConversationalAgent(name="worker", llm=llm)

        team = AgentTeam(
            agents=[worker],
            mode="supervisor",
            supervisor=supervisor,
        )
        assert team.mode == TeamMode.SUPERVISOR

    def test_team_init_accepts_enum(self) -> None:
        from tests.conftest import MockLLM
        from agent_harness.agent.conversational import ConversationalAgent

        llm = MockLLM()
        supervisor = ConversationalAgent(name="sup", llm=llm)
        worker = ConversationalAgent(name="worker", llm=llm)

        team = AgentTeam(
            agents=[worker],
            mode=TeamMode.SUPERVISOR,
            supervisor=supervisor,
        )
        assert team.mode == TeamMode.SUPERVISOR
