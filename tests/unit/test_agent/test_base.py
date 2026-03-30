"""Regression tests for BaseAgent bug fixes.

Covers:
- Agent reuse (second run() after FINISHED state)
- Usage accumulation across steps
- PlanAgent state reset on re-run
- use_long_term_memory instance flag
- Approval integration
"""
from __future__ import annotations

import pytest

from agent_harness.agent.base import StepResult
from agent_harness.agent.conversational import ConversationalAgent
from agent_harness.approval import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResult,
)
from agent_harness.context.state import AgentState
from agent_harness.core.config import HarnessConfig, LLMConfig, TracingConfig
from agent_harness.core.message import Message
from agent_harness.hooks import DefaultHooks, TracingHooks
from tests.conftest import MockLLM, MockTool


class _FailsOnceAgent(ConversationalAgent):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._should_fail = True

    async def step(self) -> StepResult:
        if self._should_fail:
            self._should_fail = False
            raise RuntimeError("intentional failure")
        return await super().step()


class TestAgentReuse:
    """Issue #1: Second run() on a finished agent must not crash."""

    @pytest.mark.asyncio
    async def test_second_run_succeeds(self) -> None:
        llm = MockLLM()
        llm.add_text_response("first answer")
        llm.add_text_response("second answer")

        agent = ConversationalAgent(name="test", llm=llm, system_prompt="")

        r1 = await agent.run("hello")
        assert r1.output == "first answer"
        assert agent.context.state.current == AgentState.FINISHED

        r2 = await agent.run("hello again")
        assert r2.output == "second answer"

    @pytest.mark.asyncio
    async def test_run_after_real_error_succeeds(self) -> None:
        llm = MockLLM()
        llm.add_text_response("recovered")

        agent = _FailsOnceAgent(name="test", llm=llm, system_prompt="")

        with pytest.raises(RuntimeError, match="intentional failure"):
            await agent.run("first")
        assert agent.context.state.current == AgentState.ERROR

        r2 = await agent.run("second")
        assert r2.output == "recovered"
        assert agent.context.state.current == AgentState.FINISHED

    @pytest.mark.asyncio
    async def test_system_prompt_not_duplicated_on_rerun(self) -> None:
        llm = MockLLM()
        llm.add_text_response("first")
        llm.add_text_response("second")

        agent = ConversationalAgent(name="test", llm=llm, system_prompt="SYS")
        await agent.run("hello")
        await agent.run("hello again")

        messages = await agent.context.short_term_memory.get_context_messages()
        system_messages = [
            msg
            for msg in messages
            if msg.role.value == "system" and (msg.content or "") == "SYS"
        ]
        assert len(system_messages) == 1

    @pytest.mark.asyncio
    async def test_system_prompt_injected_when_first_system_differs(self) -> None:
        llm = MockLLM()
        llm.add_text_response("ok")

        agent = ConversationalAgent(name="test", llm=llm, system_prompt="SYS_B")
        await agent.context.short_term_memory.add_message(Message.system("SYS_A"))

        await agent.run("hello")
        messages = await agent.context.short_term_memory.get_context_messages()
        assert any(
            msg.role.value == "system" and (msg.content or "") == "SYS_B"
            for msg in messages
        )


class TestUsageAccumulation:
    """Issue #2: AgentResult.usage must reflect actual token consumption."""

    @pytest.mark.asyncio
    async def test_usage_is_nonzero(self) -> None:
        llm = MockLLM()
        llm.add_text_response("answer")

        agent = ConversationalAgent(name="test", llm=llm, system_prompt="")
        result = await agent.run("question")

        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0
        assert result.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_usage_accumulates_across_steps(self) -> None:
        """For agents with multiple steps, usage should sum up."""
        from agent_harness.agent.react import ReActAgent
        from tests.conftest import MockTool

        llm = MockLLM()
        tool = MockTool(response="tool result")

        # Step 1: tool call → Step 2: observe + respond
        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("final answer")

        agent = ReActAgent(name="test", llm=llm, tools=[tool], system_prompt="")
        result = await agent.run("do something")

        # Two LLM calls → usage should be at least 2 × (10 prompt + 5 completion)
        assert result.usage.prompt_tokens >= 20
        assert result.usage.completion_tokens >= 10


class TestUseLongTermMemory:
    """use_long_term_memory flag on BaseAgent controls call_llm default."""

    @pytest.mark.asyncio
    async def test_long_term_memory_flag_defaults_false(self) -> None:
        """Default: use_long_term_memory is False, so long-term is not used."""
        from unittest.mock import AsyncMock, patch

        llm = MockLLM()
        llm.add_text_response("answer")

        agent = ConversationalAgent(name="test", llm=llm, system_prompt="")
        assert agent.use_long_term_memory is False

        with patch.object(
            agent.context, "build_llm_messages", new_callable=AsyncMock
        ) as mock_build:
            mock_build.return_value = []
            # Bypass the rest of run(); call call_llm directly
            await agent.call_llm()

            mock_build.assert_called_once()
            _, kwargs = mock_build.call_args
            assert kwargs["include_long_term"] is False

    @pytest.mark.asyncio
    async def test_long_term_memory_flag_true_propagates(self) -> None:
        """use_long_term_memory=True makes call_llm() use long-term by default."""
        from unittest.mock import AsyncMock, patch

        llm = MockLLM()
        llm.add_text_response("answer")

        agent = ConversationalAgent(
            name="test", llm=llm, system_prompt="", use_long_term_memory=True,
        )
        assert agent.use_long_term_memory is True

        with patch.object(
            agent.context, "build_llm_messages", new_callable=AsyncMock
        ) as mock_build:
            mock_build.return_value = []
            await agent.call_llm()

            mock_build.assert_called_once()
            _, kwargs = mock_build.call_args
            assert kwargs["include_long_term"] is True


class TestAgentConfigPropagation:
    @pytest.mark.asyncio
    async def test_agent_uses_explicit_config_when_context_not_passed(self) -> None:
        original_instance = HarnessConfig._instance
        HarnessConfig._instance = HarnessConfig(tracing=TracingConfig(enabled=True))

        llm = MockLLM()
        llm.add_text_response("answer")
        explicit_config = HarnessConfig(tracing=TracingConfig(enabled=False))

        try:
            agent = ConversationalAgent(
                name="test",
                llm=llm,
                system_prompt="",
                config=explicit_config,
            )
            assert agent.context.config is explicit_config
            assert isinstance(agent.hooks, DefaultHooks)
            assert not isinstance(agent.hooks, TracingHooks)
            assert HarnessConfig.get().tracing.enabled is True
        finally:
            HarnessConfig._instance = original_instance

    @pytest.mark.asyncio
    async def test_agent_silently_reuses_active_global_config(self) -> None:
        original_instance = HarnessConfig._instance
        active_config = HarnessConfig(tracing=TracingConfig(enabled=False))
        HarnessConfig._instance = active_config

        first_llm = MockLLM()
        first_llm.add_text_response("first")
        second_llm = MockLLM()
        second_llm.add_text_response("second")

        try:
            first_agent = ConversationalAgent(
                name="first",
                llm=first_llm,
                system_prompt="",
            )
            second_agent = ConversationalAgent(
                name="second",
                llm=second_llm,
                system_prompt="",
            )

            assert first_agent.context.config is active_config
            assert second_agent.context.config is active_config
            assert isinstance(first_agent.hooks, DefaultHooks)
            assert not isinstance(first_agent.hooks, TracingHooks)
        finally:
            HarnessConfig._instance = original_instance

    @pytest.mark.asyncio
    async def test_explicit_false_overrides_instance_flag(self) -> None:
        """Explicit use_long_term=False in call_llm() overrides the instance setting."""
        from unittest.mock import AsyncMock, patch

        llm = MockLLM()
        llm.add_text_response("answer")

        agent = ConversationalAgent(
            name="test", llm=llm, system_prompt="", use_long_term_memory=True,
        )

        with patch.object(
            agent.context, "build_llm_messages", new_callable=AsyncMock
        ) as mock_build:
            mock_build.return_value = []
            await agent.call_llm(use_long_term=False)

            mock_build.assert_called_once()
            _, kwargs = mock_build.call_args
            assert kwargs["include_long_term"] is False

    @pytest.mark.asyncio
    async def test_explicit_true_overrides_instance_default(self) -> None:
        """Explicit use_long_term=True in call_llm() works even when instance flag is False."""
        from unittest.mock import AsyncMock, patch

        llm = MockLLM()
        llm.add_text_response("answer")

        agent = ConversationalAgent(name="test", llm=llm, system_prompt="")

        with patch.object(
            agent.context, "build_llm_messages", new_callable=AsyncMock
        ) as mock_build:
            mock_build.return_value = []
            await agent.call_llm(use_long_term=True)

            mock_build.assert_called_once()
            _, kwargs = mock_build.call_args
            assert kwargs["include_long_term"] is True

    @pytest.mark.asyncio
    async def test_agent_auto_creates_llm_when_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created_llm = MockLLM()
        created_llm.add_text_response("auto-created")

        def _fake_create_llm(config: HarnessConfig | LLMConfig | None = None) -> MockLLM:
            assert isinstance(config, HarnessConfig)
            assert config.llm.model == "auto-model"
            return created_llm

        monkeypatch.setattr("agent_harness.agent.base.create_llm", _fake_create_llm)

        config = HarnessConfig(
            llm=LLMConfig(provider="openai", model="auto-model", api_key="fake")
        )
        agent = ConversationalAgent(
            name="auto",
            llm=None,
            system_prompt="",
            config=config,
        )
        result = await agent.run("hello")

        assert agent.llm is created_llm
        assert result.output == "auto-created"

    @pytest.mark.asyncio
    async def test_explicit_llm_skips_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        explicit_llm = MockLLM()
        explicit_llm.add_text_response("explicit")
        called = {"value": False}

        def _fake_create_llm(config: HarnessConfig | LLMConfig | None = None) -> MockLLM:
            called["value"] = True
            return MockLLM()

        monkeypatch.setattr("agent_harness.agent.base.create_llm", _fake_create_llm)

        config = HarnessConfig(
            llm=LLMConfig(provider="openai", model="unused-model", api_key="fake")
        )
        agent = ConversationalAgent(
            name="explicit",
            llm=explicit_llm,
            system_prompt="",
            config=config,
        )
        result = await agent.run("hello")

        assert agent.llm is explicit_llm
        assert result.output == "explicit"
        assert called["value"] is False


class _MockApprovalHandler(ApprovalHandler):
    def __init__(self, decisions: dict[str, ApprovalDecision]) -> None:
        self._decisions = decisions
        self.call_count = 0

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        self.call_count += 1
        decision = self._decisions.get(
            request.tool_call.name, ApprovalDecision.ALLOW_ONCE
        )
        return ApprovalResult(tool_call_id=request.tool_call.id, decision=decision)


class TestApprovalIntegration:
    @pytest.mark.asyncio
    async def test_no_approval_passthrough(self) -> None:
        from agent_harness.agent.react import ReActAgent

        llm = MockLLM()
        tool = MockTool(response="ok")
        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("done")

        agent = ReActAgent(name="test", llm=llm, tools=[tool], system_prompt="")
        result = await agent.run("go")
        assert result.output == "done"
        assert len(tool.call_history) == 1

    @pytest.mark.asyncio
    async def test_deny_returns_error_result(self) -> None:
        from agent_harness.agent.react import ReActAgent

        llm = MockLLM()
        tool = MockTool(response="ok")
        handler = _MockApprovalHandler({"mock_tool": ApprovalDecision.DENY})

        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("tool was denied")

        agent = ReActAgent(
            name="test", llm=llm, tools=[tool], system_prompt="",
            approval=ApprovalPolicy(), approval_handler=handler,
        )
        result = await agent.run("go")

        assert result.output == "tool was denied"
        assert len(tool.call_history) == 0
        assert handler.call_count == 1

    @pytest.mark.asyncio
    async def test_allow_session_remembered(self) -> None:
        from agent_harness.agent.react import ReActAgent

        llm = MockLLM()
        tool = MockTool(response="ok")
        handler = _MockApprovalHandler({"mock_tool": ApprovalDecision.ALLOW_SESSION})

        llm.add_tool_call_response("mock_tool", {"query": "a"})
        llm.add_tool_call_response("mock_tool", {"query": "b"})
        llm.add_text_response("done")

        agent = ReActAgent(
            name="test", llm=llm, tools=[tool], system_prompt="",
            approval=ApprovalPolicy(), approval_handler=handler,
        )
        result = await agent.run("go")

        assert result.output == "done"
        assert len(tool.call_history) == 2
        assert handler.call_count == 1  # Second call was auto-approved

    @pytest.mark.asyncio
    async def test_always_allow_skips_handler(self) -> None:
        from agent_harness.agent.react import ReActAgent

        llm = MockLLM()
        tool = MockTool(response="ok")
        handler = _MockApprovalHandler({})

        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("done")

        agent = ReActAgent(
            name="test", llm=llm, tools=[tool], system_prompt="",
            approval=ApprovalPolicy(always_allow={"mock_tool"}),
            approval_handler=handler,
        )
        await agent.run("go")

        assert handler.call_count == 0
        assert len(tool.call_history) == 1

    @pytest.mark.asyncio
    async def test_always_deny_blocks_without_handler(self) -> None:
        from agent_harness.agent.react import ReActAgent

        llm = MockLLM()
        tool = MockTool(response="ok")
        handler = _MockApprovalHandler({})

        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("denied")

        agent = ReActAgent(
            name="test", llm=llm, tools=[tool], system_prompt="",
            approval=ApprovalPolicy(always_deny={"mock_tool"}),
            approval_handler=handler,
        )
        result = await agent.run("go")

        assert result.output == "denied"
        assert handler.call_count == 0
        assert len(tool.call_history) == 0

    @pytest.mark.asyncio
    async def test_handler_failure_degrades_to_deny(self) -> None:
        from agent_harness.agent.react import ReActAgent

        class _FailingHandler(ApprovalHandler):
            async def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
                raise RuntimeError("stdin closed")

        llm = MockLLM()
        tool = MockTool(response="ok")
        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("handler failed")

        agent = ReActAgent(
            name="test", llm=llm, tools=[tool], system_prompt="",
            approval=ApprovalPolicy(), approval_handler=_FailingHandler(),
        )
        result = await agent.run("go")

        assert result.output == "handler failed"
        assert len(tool.call_history) == 0

    @pytest.mark.asyncio
    async def test_config_driven_approval(self) -> None:
        from agent_harness.core.config import ApprovalConfig

        config = HarnessConfig(
            approval=ApprovalConfig(mode="auto", always_allow=["mock_tool"]),
            tracing=TracingConfig(enabled=False),
        )
        llm = MockLLM()
        tool = MockTool(response="ok")
        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("done")

        from agent_harness.agent.react import ReActAgent

        agent = ReActAgent(
            name="test", llm=llm, tools=[tool], system_prompt="",
            config=config,
        )
        # mock_tool in always_allow → no handler needed → no prompt
        assert agent._approval is not None
        result = await agent.run("go")
        assert result.output == "done"
        assert len(tool.call_history) == 1
