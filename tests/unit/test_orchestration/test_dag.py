"""Regression tests for DAG return_exceptions fix."""
from __future__ import annotations

import pytest

from agent_harness.agent.base import AgentResult
from agent_harness.orchestration.dag import DAGNode, DAGOrchestrator
from tests.conftest import MockLLM
from agent_harness.agent.conversational import ConversationalAgent


class TestDAGErrorHandling:
    """Issue #3: One failing node must not crash the entire DAG."""

    @pytest.mark.asyncio
    async def test_failing_node_does_not_crash_batch(self) -> None:
        llm = MockLLM()
        # Agent A succeeds, Agent B fails
        llm.add_text_response("result_a")

        agent_a = ConversationalAgent(name="a", llm=llm, system_prompt="")

        # Create a node with an agent that will fail
        failing_llm = MockLLM()
        failing_llm.responses = []  # will return default
        agent_b = ConversationalAgent(name="b", llm=failing_llm, system_prompt="")

        # Both run in parallel (no dependencies)
        dag = DAGOrchestrator(nodes=[
            DAGNode(id="a", agent=agent_a),
            DAGNode(id="b", agent=agent_b),
        ])

        # Should not raise — failing node is captured
        result = await dag.run("test input")
        assert "a" in result.outputs
        assert "b" in result.outputs

    @pytest.mark.asyncio
    async def test_dependent_node_gets_error_output(self) -> None:
        """A node depending on a failed node should get error text as input."""
        llm = MockLLM()
        llm.add_text_response("result_a")  # for agent a
        llm.add_text_response("result_b")  # for agent b (default)
        llm.add_text_response("synthesized")  # for agent c

        agent_a = ConversationalAgent(name="a", llm=llm, system_prompt="")
        agent_b = ConversationalAgent(name="b", llm=llm, system_prompt="")
        agent_c = ConversationalAgent(name="c", llm=llm, system_prompt="")

        dag = DAGOrchestrator(nodes=[
            DAGNode(id="a", agent=agent_a),
            DAGNode(id="b", agent=agent_b),
            DAGNode(id="c", agent=agent_c, dependencies=["a", "b"]),
        ])

        result = await dag.run("test")
        assert "c" in result.outputs
