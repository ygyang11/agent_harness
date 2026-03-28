"""Regression tests for DAG return_exceptions fix."""
from __future__ import annotations

import pytest

from agent_harness.hooks import DefaultHooks, TracingHooks
from agent_harness.agent.base import AgentResult
from agent_harness.core.config import HarnessConfig, TracingConfig
from agent_harness.orchestration.dag import DAGNode, DAGOrchestrator
from agent_harness.core.errors import OrchestrationError
from tests.conftest import MockLLM
from agent_harness.agent.conversational import ConversationalAgent


class _FailingAgent(ConversationalAgent):
    """Agent that always raises an exception on run()."""

    async def run(self, input):
        raise RuntimeError("boom")


class _DAGSpyHooks(DefaultHooks):
    def __init__(self) -> None:
        self.started: list[str] = []
        self.ended: list[str] = []
        self.node_started: list[str] = []
        self.node_ended: list[str] = []

    async def on_dag_start(self, dag_name: str) -> None:
        self.started.append(dag_name)

    async def on_dag_end(self, dag_name: str) -> None:
        self.ended.append(dag_name)

    async def on_dag_node_start(self, node_id: str) -> None:
        self.node_started.append(node_id)

    async def on_dag_node_end(self, node_id: str) -> None:
        self.node_ended.append(node_id)


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

    @pytest.mark.asyncio
    async def test_fail_fast_true_raises_on_node_failure(self) -> None:
        """With fail_fast=True, a failing node should raise OrchestrationError."""
        llm = MockLLM()
        llm.add_text_response("result_a")

        agent_a = ConversationalAgent(name="a", llm=llm, system_prompt="")
        agent_b = _FailingAgent(name="b", llm=llm, system_prompt="")

        dag = DAGOrchestrator(
            nodes=[
                DAGNode(id="a", agent=agent_a),
                DAGNode(id="b", agent=agent_b),
            ],
            fail_fast=True,
        )

        with pytest.raises(OrchestrationError, match="DAG node 'b' failed"):
            await dag.run("test input")

    @pytest.mark.asyncio
    async def test_fail_fast_false_continues_on_node_failure(self) -> None:
        """With fail_fast=False (default), failing nodes are captured, not raised."""
        llm = MockLLM()
        llm.add_text_response("result_a")

        agent_a = ConversationalAgent(name="a", llm=llm, system_prompt="")
        agent_b = _FailingAgent(name="b", llm=llm, system_prompt="")

        dag = DAGOrchestrator(
            nodes=[
                DAGNode(id="a", agent=agent_a),
                DAGNode(id="b", agent=agent_b),
            ],
            fail_fast=False,
        )

        result = await dag.run("test input")
        assert "a" in result.outputs
        assert "b" in result.outputs
        assert result.outputs["b"].output.startswith("Error:")

    @pytest.mark.asyncio
    async def test_dag_uses_hooks_configured_in_init(self) -> None:
        llm = MockLLM()
        llm.add_text_response("result")
        agent = ConversationalAgent(name="a", llm=llm, system_prompt="")
        hooks = _DAGSpyHooks()

        dag = DAGOrchestrator(
            nodes=[DAGNode(id="a", agent=agent)],
            hooks=hooks,
        )

        result = await dag.run("test input")

        assert result.outputs["a"].output == "result"
        assert hooks.started == ["dag"]
        assert hooks.ended == ["dag"]
        assert hooks.node_started == ["a"]
        assert hooks.node_ended == ["a"]

    def test_dag_uses_explicit_config_for_hooks_resolution(self) -> None:
        llm = MockLLM()
        agent = ConversationalAgent(name="a", llm=llm, system_prompt="")
        cfg = HarnessConfig(tracing=TracingConfig(enabled=False))

        dag = DAGOrchestrator(
            nodes=[DAGNode(id="a", agent=agent)],
            config=cfg,
        )

        assert isinstance(dag.hooks, DefaultHooks)
        assert not isinstance(dag.hooks, TracingHooks)

    def test_dag_without_config_uses_active_global_config(self) -> None:
        original = HarnessConfig._instance
        HarnessConfig._instance = HarnessConfig(tracing=TracingConfig(enabled=False))
        llm = MockLLM()
        agent = ConversationalAgent(name="a", llm=llm, system_prompt="")

        try:
            dag = DAGOrchestrator(nodes=[DAGNode(id="a", agent=agent)])
            assert isinstance(dag.hooks, DefaultHooks)
            assert not isinstance(dag.hooks, TracingHooks)
        finally:
            HarnessConfig._instance = original
