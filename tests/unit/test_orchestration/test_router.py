"""Tests for router execution semantics."""
from __future__ import annotations

import pytest

from agent_harness.agent.conversational import ConversationalAgent
from agent_harness.hooks import DefaultHooks, TracingHooks
from agent_harness.core.config import HarnessConfig, TracingConfig
from agent_harness.orchestration.router import AgentRouter, Route

from tests.conftest import MockLLM


class _SpyHooks(DefaultHooks):
    def __init__(self) -> None:
        self.started: list[str] = []

    async def on_run_start(self, agent_name: str, input_text: str) -> None:
        self.started.append(agent_name)


class TestAgentRouter:
    @pytest.mark.asyncio
    async def test_router_does_not_override_agent_hooks(self) -> None:
        llm = MockLLM()
        llm.add_text_response("42")
        agent_hooks = _SpyHooks()
        math_agent = ConversationalAgent(
            name="math",
            llm=llm,
            system_prompt="",
            hooks=agent_hooks,
        )
        router_hooks = _SpyHooks()
        router = AgentRouter(
            routes=[Route(agent=math_agent, name="math", condition=r"calc|math|add|\d+\s*\+")],
            hooks=router_hooks,
        )

        result = await router.run("Please calc 6 * 7")

        assert result.output == "42"
        assert agent_hooks.started == ["math"]
        assert router_hooks.started == []
        assert math_agent.hooks is agent_hooks

    def test_router_uses_explicit_config_for_hooks_resolution(self) -> None:
        llm = MockLLM()
        agent = ConversationalAgent(name="math", llm=llm, system_prompt="")
        cfg = HarnessConfig(tracing=TracingConfig(enabled=False))

        router = AgentRouter(
            routes=[Route(agent=agent, name="math", condition=r"math")],
            config=cfg,
        )

        assert isinstance(router.hooks, DefaultHooks)
        assert not isinstance(router.hooks, TracingHooks)

    def test_router_without_config_uses_active_global_config(self) -> None:
        original = HarnessConfig._instance
        HarnessConfig._instance = HarnessConfig(tracing=TracingConfig(enabled=False))
        llm = MockLLM()
        agent = ConversationalAgent(name="math", llm=llm, system_prompt="")

        try:
            router = AgentRouter(routes=[Route(agent=agent, name="math", condition=r"math")])
            assert isinstance(router.hooks, DefaultHooks)
            assert not isinstance(router.hooks, TracingHooks)
        finally:
            HarnessConfig._instance = original
