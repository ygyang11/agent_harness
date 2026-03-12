"""Tests for TracingHooks — span creation and lifecycle management."""
from __future__ import annotations

import pytest

from agent_harness.tracing.hooks import TracingHooks
from agent_harness.tracing.tracer import Span


class TestTracingHooksLifecycle:
    @pytest.mark.asyncio
    async def test_run_lifecycle_creates_and_finishes_root_span(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("test_agent", "hello")
        assert hooks._root_span is not None
        assert hooks._root_span.kind == "agent"
        assert hooks._root_span.name == "agent.test_agent"

        await hooks.on_run_end("test_agent", "goodbye")
        assert hooks._root_span is None
        assert len(hooks._all_spans) == 1
        assert hooks._all_spans[0].end_time is not None

    @pytest.mark.asyncio
    async def test_step_spans_are_children(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("agent", "input")

        await hooks.on_step_start("agent", 1)
        assert hooks._current_step_span is not None
        step_span = hooks._current_step_span
        assert step_span.parent_span_id == hooks._root_span.span_id

        await hooks.on_step_end("agent", 1)
        assert hooks._current_step_span is None
        assert step_span in hooks._all_spans

        await hooks.on_run_end("agent", "done")

    @pytest.mark.asyncio
    async def test_llm_and_tool_events_recorded(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("agent", "input")

        await hooks.on_step_start("agent", 1)
        await hooks.on_llm_call("agent", [{"role": "user", "content": "hi"}])
        await hooks.on_tool_call("agent", type("TC", (), {"name": "search"})())
        await hooks.on_tool_result("agent", type("TR", (), {"output": "result"})())
        await hooks.on_step_end("agent", 1)

        step_span = hooks._all_spans[0]
        event_names = [e.name for e in step_span.events]
        assert "llm_call" in event_names
        assert "tool_call" in event_names
        assert "tool_result" in event_names

        await hooks.on_run_end("agent", "done")

    @pytest.mark.asyncio
    async def test_error_marks_span(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("agent", "input")

        error = ValueError("test error")
        await hooks.on_error("agent", error)

        assert hooks._root_span is None
        assert len(hooks._all_spans) == 1
        assert hooks._all_spans[0].status == "error"
        assert "test error" in hooks._all_spans[0].error_message


class TestTracingHooksOrchestration:
    @pytest.mark.asyncio
    async def test_pipeline_spans(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_pipeline_start("my_pipeline")
        assert len(hooks._span_stack) == 1
        assert "pipeline" in hooks._span_stack[0].name

        await hooks.on_pipeline_end("my_pipeline")
        assert len(hooks._all_spans) == 1

    @pytest.mark.asyncio
    async def test_team_spans(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_team_start("my_team", "supervisor")
        assert hooks._span_stack[-1].attributes["mode"] == "supervisor"

        await hooks.on_team_end("my_team", "supervisor")
        assert len(hooks._all_spans) == 1

    @pytest.mark.asyncio
    async def test_dag_spans_with_nodes(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_dag_start("my_dag")

        await hooks.on_dag_node_start("node_1")
        await hooks.on_dag_node_end("node_1")

        await hooks.on_dag_end("my_dag")
        assert len(hooks._all_spans) == 2  # dag_node + dag
