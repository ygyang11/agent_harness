"""Tests for agent_harness.tool.executor — ToolExecutor execution, errors, timeout, batch."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_harness.core.config import HarnessConfig, ToolConfig
from agent_harness.core.message import ToolCall
from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.tool.executor import ToolExecutor
from agent_harness.tool.registry import ToolRegistry

from tests.conftest import MockTool


class _SlowTool(BaseTool):
    """Tool that sleeps to test timeout."""

    def __init__(self, delay: float = 5.0) -> None:
        super().__init__(name="slow_tool", description="A slow tool")
        self.delay = delay

    async def execute(self, **kwargs: Any) -> str:
        await asyncio.sleep(self.delay)
        return "done"


class _FailingTool(BaseTool):
    """Tool that always raises."""

    def __init__(self) -> None:
        super().__init__(name="fail_tool", description="Always fails")

    async def execute(self, **kwargs: Any) -> str:
        raise RuntimeError("tool exploded")


def _make_executor(*tools: BaseTool, config: ToolConfig | None = None) -> ToolExecutor:
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    return ToolExecutor(registry, config=config)


class TestToolExecutorSuccess:
    @pytest.mark.asyncio
    async def test_successful_execution(self) -> None:
        mock = MockTool(response="hello world")
        executor = _make_executor(mock)
        tc = ToolCall(name="mock_tool", arguments={"query": "test"})

        result = await executor.execute(tc)
        assert result.content == "hello world"
        assert result.is_error is False
        assert result.tool_call_id == tc.id
        assert mock.call_history == [{"query": "test"}]

    @pytest.mark.asyncio
    async def test_execution_with_no_args(self) -> None:
        mock = MockTool(response="ok")
        executor = _make_executor(mock)
        tc = ToolCall(name="mock_tool", arguments={})
        result = await executor.execute(tc)
        assert result.content == "ok"

    @pytest.mark.asyncio
    async def test_accepts_harness_config(self) -> None:
        mock = MockTool(response="ok")
        cfg = HarnessConfig(tool=ToolConfig(max_concurrency=2, default_timeout=0.2))
        executor = _make_executor(mock, config=cfg)

        tc = ToolCall(name="mock_tool", arguments={})
        result = await executor.execute(tc)

        assert executor.config.max_concurrency == 2
        assert executor.config.default_timeout == 0.2
        assert result.content == "ok"


class TestToolExecutorErrors:
    @pytest.mark.asyncio
    async def test_tool_not_found(self) -> None:
        executor = _make_executor()  # empty registry
        tc = ToolCall(name="missing_tool", arguments={})
        result = await executor.execute(tc)
        assert result.is_error is True
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_tool_raises_exception(self) -> None:
        executor = _make_executor(_FailingTool())
        tc = ToolCall(name="fail_tool", arguments={})
        result = await executor.execute(tc)
        assert result.is_error is True
        assert "tool exploded" in result.content.lower() or "error" in result.content.lower()


class TestToolExecutorTimeout:
    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        slow = _SlowTool(delay=10.0)
        executor = _make_executor(slow)
        tc = ToolCall(name="slow_tool", arguments={})
        result = await executor.execute(tc, timeout=0.1)
        assert result.is_error is True
        assert "timed out" in result.content.lower()

    @pytest.mark.asyncio
    async def test_default_timeout_from_config(self) -> None:
        slow = _SlowTool(delay=10.0)
        config = ToolConfig(default_timeout=0.1)
        executor = _make_executor(slow, config=config)
        tc = ToolCall(name="slow_tool", arguments={})
        result = await executor.execute(tc)
        assert result.is_error is True
        assert "timed out" in result.content.lower()


class TestToolExecutorBatch:
    @pytest.mark.asyncio
    async def test_batch_execution(self) -> None:
        mock = MockTool(response="batch_result")
        executor = _make_executor(mock)

        calls = [
            ToolCall(name="mock_tool", arguments={"query": "a"}),
            ToolCall(name="mock_tool", arguments={"query": "b"}),
            ToolCall(name="mock_tool", arguments={"query": "c"}),
        ]
        results = await executor.execute_batch(calls)

        assert len(results) == 3
        assert all(r.content == "batch_result" for r in results)
        assert all(r.is_error is False for r in results)
        for call, result in zip(calls, results):
            assert result.tool_call_id == call.id

    @pytest.mark.asyncio
    async def test_batch_empty(self) -> None:
        executor = _make_executor()
        results = await executor.execute_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_with_mixed_results(self) -> None:
        mock = MockTool(response="ok")
        executor = _make_executor(mock)
        calls = [
            ToolCall(name="mock_tool", arguments={}),
            ToolCall(name="nonexistent", arguments={}),
        ]
        results = await executor.execute_batch(calls)
        assert len(results) == 2
        assert results[0].is_error is False
        assert results[1].is_error is True
