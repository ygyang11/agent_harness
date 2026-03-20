"""Tests for TracingHooks — span creation and lifecycle management."""
from __future__ import annotations

from collections.abc import Mapping

import pytest

from agent_harness.agent.hooks import DefaultHooks, TracingHooks, resolve_hooks
from agent_harness.core.config import HarnessConfig, TracingConfig
from agent_harness.core.message import ToolCall
from agent_harness.tracing.tracer import Span
from agent_harness.utils.token_counter import count_tokens


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
        assert hooks._depth == 0
        assert hooks._all_spans[0].end_time is not None

    @pytest.mark.asyncio
    async def test_step_spans_are_children(self) -> None:
        from agent_harness.agent.hooks import _active_step_span

        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("agent", "input")

        await hooks.on_step_start("agent", 1)
        step_span = _active_step_span.get(None)
        assert step_span is not None
        assert step_span.parent_span_id == hooks._root_span.span_id

        await hooks.on_step_end("agent", 1)
        assert _active_step_span.get(None) is None
        assert step_span in hooks._all_spans

        await hooks.on_run_end("agent", "done")

    @pytest.mark.asyncio
    async def test_llm_and_tool_events_recorded(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("agent", "input")

        await hooks.on_step_start("agent", 1)
        await hooks.on_llm_call("agent", [{"role": "user", "content": "hi"}])
        await hooks.on_tool_call("agent", ToolCall(name="search", arguments={"query": "hi"}))
        await hooks.on_tool_result("agent", type("TR", (), {"output": "result"})())
        await hooks.on_step_end("agent", 1)

        step_span = hooks._all_spans[0]
        event_names = [e.name for e in step_span.events]
        assert "llm_call" in event_names
        assert "tool_call" in event_names
        assert "tool_result" in event_names

        await hooks.on_run_end("agent", "done")

    @pytest.mark.asyncio
    async def test_text_attributes_truncated_by_tokens(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        long_text = "signal " * 600

        await hooks.on_run_start("agent", long_text)
        assert hooks._root_span is not None
        run_input = hooks._root_span.attributes["input"]
        assert count_tokens(run_input) <= 64

        await hooks.on_tool_result("agent", type("TR", (), {"output": long_text})())
        assert hooks._root_span.events
        last_event = hooks._root_span.events[-1]
        assert last_event.name == "tool_result"
        assert count_tokens(last_event.attributes["output"]) <= 64

        await hooks.on_run_end("agent", long_text)
        run_span = hooks._all_spans[0]
        assert count_tokens(run_span.attributes["output"]) <= 64

    @pytest.mark.asyncio
    async def test_run_start_prints_start_marker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        start_calls: list[tuple[str, str, int, dict[str, object] | None]] = []

        def capture_start(
            *,
            kind: str,
            name: str,
            indent: int = 0,
            attributes: Mapping[str, object] | None = None,
        ) -> None:
            payload = dict(attributes) if attributes is not None else None
            start_calls.append((kind, name, indent, payload))

        monkeypatch.setattr(hooks._console, "print_start", capture_start)

        await hooks.on_run_start("agent", "hello world")

        assert len(start_calls) == 1
        kind, name, indent, attrs = start_calls[0]
        assert kind == "agent"
        assert name == "agent.agent"
        assert indent == 0
        assert attrs is not None
        assert attrs.get("input") == "hello world"

    @pytest.mark.asyncio
    async def test_finish_span_uses_runtime_stack_indent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        export_calls: list[tuple[str, int]] = []

        def capture_export(span: Span, *, indent: int) -> None:
            export_calls.append((span.name, indent))

        monkeypatch.setattr(hooks._console, "export_one", capture_export)

        await hooks.on_run_start("agent", "input")
        await hooks.on_step_start("agent", 1)
        await hooks.on_step_end("agent", 1)
        await hooks.on_run_end("agent", "done")

        assert ("step.1", 1) in export_calls
        assert ("agent.agent", 0) in export_calls

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

    @pytest.mark.asyncio
    async def test_tool_call_records_sanitized_args(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("agent", "input")
        await hooks.on_step_start("agent", 1)

        await hooks.on_tool_call(
            "agent",
            ToolCall(
                name="web_search",
                arguments={
                    "query": "renewable energy trend 2025",
                    "api_key": "secret-token",
                    "max_results": 3,
                },
            ),
        )
        await hooks.on_step_end("agent", 1)

        step_span = hooks._all_spans[0]
        event = next(e for e in step_span.events if e.name == "tool_call")
        assert event.attributes["tool"] == "web_search"
        args = event.attributes["args"]
        assert isinstance(args, dict)
        assert args["api_key"] == "[REDACTED]"
        assert args["query"] == "renewable energy trend 2025"
        assert args["max_results"] == "3"

        await hooks.on_run_end("agent", "done")

    @pytest.mark.asyncio
    async def test_tool_call_argument_value_truncated_by_tokens(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("agent", "input")
        await hooks.on_step_start("agent", 1)

        long_query = "signal " * 400
        await hooks.on_tool_call(
            "agent",
            ToolCall(name="web_search", arguments={"query": long_query}),
        )
        await hooks.on_step_end("agent", 1)

        step_span = hooks._all_spans[0]
        event = next(e for e in step_span.events if e.name == "tool_call")
        args = event.attributes["args"]
        assert isinstance(args, dict)
        assert count_tokens(args["query"]) <= 20

        await hooks.on_run_end("agent", "done")


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

    @pytest.mark.asyncio
    async def test_orchestration_starts_print_markers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        start_calls: list[str] = []

        def capture_start(
            *,
            kind: str,
            name: str,
            indent: int = 0,
            attributes: Mapping[str, object] | None = None,
        ) -> None:
            del kind, indent, attributes
            start_calls.append(name)

        monkeypatch.setattr(hooks._console, "print_start", capture_start)

        await hooks.on_pipeline_start("pipe")
        await hooks.on_pipeline_end("pipe")

        await hooks.on_dag_start("dag")
        await hooks.on_dag_node_start("node_1")
        await hooks.on_dag_node_end("node_1")
        await hooks.on_dag_end("dag")

        await hooks.on_team_start("team", "supervisor")
        await hooks.on_team_end("team", "supervisor")

        assert "pipeline.pipe" in start_calls
        assert "dag.dag" in start_calls
        assert "dag_node.node_1" in start_calls
        assert "team.team" in start_calls

    @pytest.mark.asyncio
    async def test_nested_tracing_preserves_outer_spans(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_pipeline_start("pipe")

        await hooks.on_run_start("agent1", "input1")
        await hooks.on_run_end("agent1", "output1")
        assert hooks._depth == 1

        await hooks.on_run_start("agent2", "input2")
        await hooks.on_run_end("agent2", "output2")
        assert hooks._depth == 1

        await hooks.on_pipeline_end("pipe")
        assert hooks._depth == 0
        assert len(hooks._all_spans) == 3
        names = {span.name for span in hooks._all_spans}
        assert "pipeline.pipe" in names
        assert "agent.agent1" in names
        assert "agent.agent2" in names

    @pytest.mark.asyncio
    async def test_nested_error_does_not_break_outer_orchestration_span(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_pipeline_start("pipe")
        await hooks.on_run_start("agent1", "input1")
        await hooks.on_step_start("agent1", 1)

        await hooks.on_error("agent1", RuntimeError("boom"))
        assert hooks._depth == 1
        assert [span.name for span in hooks._span_stack] == ["pipeline.pipe"]

        await hooks.on_pipeline_end("pipe")
        assert hooks._depth == 0
        assert hooks._span_stack == []

        names = {span.name for span in hooks._all_spans}
        assert "pipeline.pipe" in names
        assert "agent.agent1" in names
        assert "step.1" in names


class TestResolveHooks:
    def test_disabled_tracing_returns_default_even_with_explicit_hooks(self) -> None:
        explicit = TracingHooks(trace_dir="/tmp/explicit")
        cfg = HarnessConfig(tracing=TracingConfig(enabled=False))

        resolved = resolve_hooks(explicit, cfg)

        assert isinstance(resolved, DefaultHooks)
        assert not isinstance(resolved, TracingHooks)

    def test_enabled_tracing_uses_explicit_hooks(self) -> None:
        explicit = DefaultHooks()
        cfg = HarnessConfig(tracing=TracingConfig(enabled=True))

        resolved = resolve_hooks(explicit, cfg)

        assert resolved is explicit

    def test_enabled_tracing_autocreates_tracing_hooks(self) -> None:
        cfg = HarnessConfig(tracing=TracingConfig(enabled=True, export_path="/tmp/from_config"))

        resolved = resolve_hooks(None, cfg)

        assert isinstance(resolved, TracingHooks)
        assert resolved._trace_dir == "/tmp/from_config"

    def test_enabled_tracing_reuses_cached_hooks(self) -> None:
        cfg = HarnessConfig(tracing=TracingConfig(enabled=True, export_path="/tmp/from_config"))

        first = resolve_hooks(None, cfg)
        second = resolve_hooks(None, cfg)

        assert isinstance(first, TracingHooks)
        assert first is second


class TestOrchestrationSpanKind:
    @pytest.mark.asyncio
    async def test_orchestration_spans_use_orchestration_kind(self) -> None:
        """Pipeline, DAG, DAG node, and team spans should have kind='orchestration'."""
        hooks = TracingHooks(trace_dir="/tmp/test_traces")

        await hooks.on_pipeline_start("pipe")
        assert hooks._span_stack[-1].kind == "orchestration"
        await hooks.on_pipeline_end("pipe")

        await hooks.on_dag_start("dag")
        assert hooks._span_stack[-1].kind == "orchestration"
        await hooks.on_dag_node_start("n1")
        assert hooks._span_stack[-1].kind == "orchestration"
        await hooks.on_dag_node_end("n1")
        await hooks.on_dag_end("dag")

        await hooks.on_team_start("team", "supervisor")
        assert hooks._span_stack[-1].kind == "orchestration"
        await hooks.on_team_end("team", "supervisor")


class TestParallelDAGNodes:
    @pytest.mark.asyncio
    async def test_parallel_dag_nodes_are_siblings(self) -> None:
        """Parallel DAG nodes should share the DAG span as parent and have equal depth."""
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_dag_start("dag")
        dag_span = hooks._dag_span_stack[-1]

        await hooks.on_dag_node_start("a")
        span_a = hooks._dag_node_span_stack[-1]
        await hooks.on_dag_node_start("b")
        span_b = hooks._dag_node_span_stack[-1]

        assert span_a.parent_span_id == dag_span.span_id
        assert span_b.parent_span_id == dag_span.span_id
        assert hooks._span_depth_map[span_a.span_id] == hooks._span_depth_map[span_b.span_id]

        await hooks.on_dag_node_end("b")
        await hooks.on_dag_node_end("a")
        await hooks.on_dag_end("dag")

    @pytest.mark.asyncio
    async def test_agent_inside_dag_node_uses_node_as_parent(self) -> None:
        """An agent started inside a DAG node should use the node span as parent."""
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_dag_start("dag")

        await hooks.on_dag_node_start("n1")
        node_span = hooks._dag_node_span_stack[-1]

        await hooks.on_run_start("agent1", "hello")
        agent_span = hooks._run_span_stack[-1]
        assert agent_span.parent_span_id == node_span.span_id

        await hooks.on_run_end("agent1", "done")
        await hooks.on_dag_node_end("n1")
        await hooks.on_dag_end("dag")


class TestParallelTeamWorkers:
    @pytest.mark.asyncio
    async def test_parallel_team_workers_are_siblings(self) -> None:
        """Parallel team workers should share the team span as parent."""
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_team_start("team", "supervisor")
        team_span = hooks._team_span_stack[-1]

        await hooks.on_run_start("worker_a", "task a")
        span_a = hooks._run_span_stack[-1]
        await hooks.on_run_start("worker_b", "task b")
        span_b = hooks._run_span_stack[-1]

        assert span_a.parent_span_id == team_span.span_id
        assert span_b.parent_span_id == team_span.span_id
        assert hooks._span_depth_map[span_a.span_id] == hooks._span_depth_map[span_b.span_id]

        await hooks.on_run_end("worker_b", "done b")
        await hooks.on_run_end("worker_a", "done a")
        await hooks.on_team_end("team", "supervisor")

    @pytest.mark.asyncio
    async def test_pipeline_agents_use_pipeline_as_parent(self) -> None:
        """Agents inside a pipeline should use the pipeline span as parent."""
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_pipeline_start("pipe")
        pipe_span = hooks._pipeline_span_stack[-1]

        await hooks.on_run_start("agent1", "hello")
        agent_span = hooks._run_span_stack[-1]
        assert agent_span.parent_span_id == pipe_span.span_id

        await hooks.on_run_end("agent1", "done")
        await hooks.on_pipeline_end("pipe")


class TestParallelStepSpans:
    @pytest.mark.asyncio
    async def test_parallel_step_spans_tracked_independently(self) -> None:
        """Concurrent step spans in separate tasks should not overwrite each other."""
        import asyncio

        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_team_start("team", "supervisor")

        async def run_worker(name: str) -> None:
            await hooks.on_run_start(name, f"task {name}")
            await hooks.on_step_start(name, 1)
            await asyncio.sleep(0.01)
            await hooks.on_step_end(name, 1)
            await hooks.on_run_end(name, f"done {name}")

        await asyncio.gather(run_worker("a"), run_worker("b"))
        await hooks.on_team_end("team", "supervisor")

        step_spans = [s for s in hooks._all_spans if s.name == "step.1"]
        agent_spans = [s for s in hooks._all_spans if s.kind == "agent"]
        assert len(step_spans) == 2
        assert len(agent_spans) == 2

    @pytest.mark.asyncio
    async def test_parallel_workers_all_run_spans_collected(self) -> None:
        """All parallel worker agent spans should appear in _all_spans."""
        import asyncio

        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_team_start("team", "supervisor")

        async def run_worker(name: str) -> None:
            await hooks.on_run_start(name, f"task {name}")
            await hooks.on_step_start(name, 1)
            await asyncio.sleep(0.01)
            await hooks.on_step_end(name, 1)
            await hooks.on_run_end(name, f"done {name}")

        await asyncio.gather(
            run_worker("a"), run_worker("b"), run_worker("c"),
        )
        await hooks.on_team_end("team", "supervisor")

        agent_spans = [s for s in hooks._all_spans if s.kind == "agent"]
        agent_names = {s.name for s in agent_spans}
        assert len(agent_spans) == 3
        assert agent_names == {"agent.a", "agent.b", "agent.c"}

    @pytest.mark.asyncio
    async def test_parallel_step_parent_is_own_run_span(self) -> None:
        """Each parallel step span should be a child of its own agent run span."""
        import asyncio

        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_team_start("team", "supervisor")

        async def run_worker(name: str) -> None:
            await hooks.on_run_start(name, f"task {name}")
            await hooks.on_step_start(name, 1)
            await asyncio.sleep(0.01)
            await hooks.on_step_end(name, 1)
            await hooks.on_run_end(name, f"done {name}")

        await asyncio.gather(run_worker("a"), run_worker("b"))
        await hooks.on_team_end("team", "supervisor")

        agent_spans = {s.name: s for s in hooks._all_spans if s.kind == "agent"}
        step_spans = [s for s in hooks._all_spans if s.name == "step.1"]

        assert len(step_spans) == 2
        for step_span in step_spans:
            assert step_span.parent_span_id in {s.span_id for s in agent_spans.values()}
