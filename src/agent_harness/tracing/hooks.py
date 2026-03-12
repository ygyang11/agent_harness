"""TracingHooks — automatic span management via agent lifecycle hooks.

TracingHooks extends DefaultHooks to create and manage Span objects at
each stage of agent execution.  It also provides orchestration-level
methods (on_pipeline_*, on_dag_*, on_team_*) that are called directly
by the Pipeline, DAG, and Team orchestrators when they detect that the
hooks instance supports them.

Runtime output uses the ConsoleExporter (print-based, not logging) for
real-time visibility, and the JsonFileExporter for persistent traces
(file path based on timestamp).
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime
from typing import Any, TYPE_CHECKING

from agent_harness.agent.hooks import DefaultHooks
from agent_harness.tracing.tracer import Span, SpanEvent
from agent_harness.tracing.exporters.console import ConsoleExporter
from agent_harness.tracing.exporters.json_file import JsonFileExporter

if TYPE_CHECKING:
    from agent_harness.core.message import Message, ToolCall, ToolResult


class TracingHooks(DefaultHooks):
    """Hooks that automatically create and manage trace spans.

    Each ``on_run_start`` creates a *root* agent span.  Steps, LLM calls,
    tool calls, and orchestration boundaries produce child spans / events
    attached to the active root.

    On ``on_run_end`` (or ``on_error``), accumulated spans are exported to
    both console (real-time ``print``) and a timestamp-named JSON Lines
    file under ``./traces/``.
    """

    def __init__(
        self,
        *,
        trace_dir: str = "./traces",
        console_color: bool = True,
    ) -> None:
        self._trace_dir = trace_dir
        self._console = ConsoleExporter(stream=sys.stderr, color=console_color)

        # Span tracking
        self._trace_id: str = ""
        self._root_span: Span | None = None
        self._current_step_span: Span | None = None
        self._span_stack: list[Span] = []
        self._all_spans: list[Span] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_span(self, name: str, kind: str = "internal", **attrs: Any) -> Span:
        parent_id = self._span_stack[-1].span_id if self._span_stack else None
        span = Span(
            trace_id=self._trace_id,
            parent_span_id=parent_id,
            name=name,
            kind=kind,
            attributes=attrs,
        )
        self._span_stack.append(span)
        return span

    def _finish_span(self, span: Span) -> None:
        span.finish()
        self._all_spans.append(span)
        if self._span_stack and self._span_stack[-1] is span:
            self._span_stack.pop()
        # Real-time console output
        self._console.export([span])

    def _export_all(self) -> None:
        """Flush all collected spans to JSON file."""
        if not self._all_spans:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exporter = JsonFileExporter(path=f"{self._trace_dir}/{ts}.jsonl")
        exporter.export(self._all_spans)

    # ------------------------------------------------------------------
    # AgentHooks — standard lifecycle
    # ------------------------------------------------------------------

    async def on_run_start(self, agent_name: str, input_text: str) -> None:
        self._trace_id = uuid.uuid4().hex
        self._all_spans = []
        self._span_stack = []
        self._root_span = self._new_span(
            f"agent.{agent_name}", kind="agent", input=input_text[:200],
        )

    async def on_step_start(self, agent_name: str, step: int) -> None:
        self._current_step_span = self._new_span(
            f"step.{step}", kind="internal", agent=agent_name,
        )

    async def on_step_end(self, agent_name: str, step: int) -> None:
        if self._current_step_span:
            self._finish_span(self._current_step_span)
            self._current_step_span = None

    async def on_llm_call(self, agent_name: str, messages: list[Any]) -> None:
        active = self._span_stack[-1] if self._span_stack else self._root_span
        if active:
            active.add_event(
                "llm_call",
                agent=agent_name,
                message_count=len(messages),
            )

    async def on_tool_call(self, agent_name: str, tool_call: Any) -> None:
        active = self._span_stack[-1] if self._span_stack else self._root_span
        if active:
            name = getattr(tool_call, "name", str(tool_call))
            active.add_event("tool_call", agent=agent_name, tool=name)

    async def on_tool_result(self, agent_name: str, result: Any) -> None:
        active = self._span_stack[-1] if self._span_stack else self._root_span
        if active:
            output = str(getattr(result, "output", result))
            active.add_event(
                "tool_result",
                agent=agent_name,
                output=output[:200],
            )

    async def on_error(self, agent_name: str, error: Exception) -> None:
        if self._root_span:
            self._root_span.set_error(error)
            self._finish_span(self._root_span)
            self._root_span = None
        self._export_all()

    async def on_run_end(self, agent_name: str, output: str) -> None:
        if self._root_span:
            self._root_span.attributes["output"] = output[:200]
            self._finish_span(self._root_span)
            self._root_span = None
        self._export_all()

    # ------------------------------------------------------------------
    # Orchestration-level extensions
    # ------------------------------------------------------------------

    async def on_pipeline_start(self, pipeline_name: str) -> None:
        span = self._new_span(f"pipeline.{pipeline_name}", kind="internal")
        span.attributes["orchestration"] = "pipeline"

    async def on_pipeline_end(self, pipeline_name: str) -> None:
        if self._span_stack:
            self._finish_span(self._span_stack[-1])

    async def on_dag_start(self, dag_name: str) -> None:
        span = self._new_span(f"dag.{dag_name}", kind="internal")
        span.attributes["orchestration"] = "dag"

    async def on_dag_end(self, dag_name: str) -> None:
        if self._span_stack:
            self._finish_span(self._span_stack[-1])

    async def on_dag_node_start(self, node_id: str) -> None:
        self._new_span(f"dag_node.{node_id}", kind="internal")

    async def on_dag_node_end(self, node_id: str) -> None:
        if self._span_stack:
            self._finish_span(self._span_stack[-1])

    async def on_team_start(self, team_name: str, mode: str) -> None:
        span = self._new_span(f"team.{team_name}", kind="internal")
        span.attributes["orchestration"] = "team"
        span.attributes["mode"] = mode

    async def on_team_end(self, team_name: str, mode: str) -> None:
        if self._span_stack:
            self._finish_span(self._span_stack[-1])
