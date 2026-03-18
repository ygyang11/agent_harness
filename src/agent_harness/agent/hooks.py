"""Agent lifecycle hooks for extensible behavior."""
from __future__ import annotations

import sys
import uuid
from datetime import datetime
from typing import Any, TYPE_CHECKING

from agent_harness.core.config import HarnessConfig
from agent_harness.tracing.tracer import Span, SpanEvent
from agent_harness.tracing.exporters.console import ConsoleExporter
from agent_harness.tracing.exporters.json_file import JsonFileExporter
from agent_harness.utils.token_counter import truncate_text_by_tokens

if TYPE_CHECKING:
    from agent_harness.core.message import Message, ToolCall, ToolResult


_TRACE_TEXT_MAX_TOKENS = 64
_TRACE_TOOL_ARG_MAX_TOKENS = 20
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "token",
    "authorization",
    "password",
    "secret",
    "key",
}


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _SENSITIVE_KEYS or lowered.endswith("_key")


def _truncate_tool_arg(value: object) -> str:
    return truncate_text_by_tokens(
        str(value),
        max_tokens=_TRACE_TOOL_ARG_MAX_TOKENS,
        suffix="...",
    )


def _summarize_tool_arguments(arguments: dict[str, object]) -> dict[str, str]:
    summarized: dict[str, str] = {}
    for key, value in arguments.items():
        summarized[key] = "[REDACTED]" if _is_sensitive_key(key) else _truncate_tool_arg(value)
    return summarized


def _summarize_tool_call(tool_call: ToolCall) -> dict[str, object]:
    return {
        "tool": tool_call.name,
        "args": _summarize_tool_arguments(tool_call.arguments),
    }


class DefaultHooks:
    """No-op base hooks. Subclass to add custom behavior."""

    async def on_run_start(self, agent_name: str, input_text: str) -> None:
        pass

    async def on_step_start(self, agent_name: str, step: int) -> None:
        pass

    async def on_llm_call(self, agent_name: str, messages: list[Any]) -> None:
        pass

    async def on_tool_call(self, agent_name: str, tool_call: ToolCall) -> None:
        pass

    async def on_tool_result(self, agent_name: str, result: Any) -> None:
        pass

    async def on_step_end(self, agent_name: str, step: int) -> None:
        pass

    async def on_run_end(self, agent_name: str, output: str) -> None:
        pass

    async def on_error(self, agent_name: str, error: Exception) -> None:
        pass


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
        self._run_span_stack: list[Span] = []
        self._pipeline_span_stack: list[Span] = []
        self._dag_span_stack: list[Span] = []
        self._team_span_stack: list[Span] = []
        self._dag_node_span_stack: list[Span] = []
        self._all_spans: list[Span] = []
        self._depth = 0

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
        indent = self._span_stack.index(span) if span in self._span_stack else 0
        span.finish()
        self._all_spans.append(span)
        if self._span_stack and self._span_stack[-1] is span:
            self._span_stack.pop()
        # Real-time console output
        self._console.export_one(span, indent=indent)

    def _export_all(self) -> None:
        """Flush all collected spans to JSON file."""
        if not self._all_spans:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exporter = JsonFileExporter(path=f"{self._trace_dir}/{ts}.jsonl")
        exporter.export(self._all_spans)

    def _finish_until(self, target: Span) -> None:
        if target not in self._span_stack:
            return
        while self._span_stack:
            span = self._span_stack[-1]
            self._finish_span(span)
            if span is target:
                break

    def _begin_execution(self) -> None:
        self._depth += 1
        if self._depth == 1:
            self._trace_id = uuid.uuid4().hex
            self._root_span = None
            self._current_step_span = None
            self._span_stack = []
            self._run_span_stack = []
            self._pipeline_span_stack = []
            self._dag_span_stack = []
            self._team_span_stack = []
            self._dag_node_span_stack = []
            self._all_spans = []

    def _end_execution(self) -> None:
        if self._depth == 0:
            return
        self._depth -= 1
        if self._depth == 0:
            self._export_all()

    # ------------------------------------------------------------------
    # Standard lifecycle hooks
    # ------------------------------------------------------------------

    async def on_run_start(self, agent_name: str, input_text: str) -> None:
        self._begin_execution()
        truncated_input = truncate_text_by_tokens(
            input_text,
            max_tokens=_TRACE_TEXT_MAX_TOKENS,
            suffix="",
        )
        run_span = self._new_span(
            f"agent.{agent_name}",
            kind="agent",
            input=truncated_input,
        )
        self._run_span_stack.append(run_span)
        self._root_span = run_span
        self._console.print_start(
            kind=run_span.kind,
            name=run_span.name,
            indent=max(len(self._span_stack) - 1, 0),
            attributes={"input": truncated_input},
        )

    async def on_step_start(self, agent_name: str, step: int) -> None:
        self._current_step_span = self._new_span(
            f"step.{step}", kind="internal", agent=agent_name,
        )

    async def on_step_end(self, agent_name: str, step: int) -> None:
        if self._current_step_span:
            self._finish_until(self._current_step_span)
            self._current_step_span = None

    async def on_llm_call(self, agent_name: str, messages: list[Any]) -> None:
        active = self._span_stack[-1] if self._span_stack else self._root_span
        if active:
            active.add_event(
                "llm_call",
                agent=agent_name,
                message_count=len(messages),
            )

    async def on_tool_call(self, agent_name: str, tool_call: ToolCall) -> None:
        active = self._span_stack[-1] if self._span_stack else self._root_span
        if active:
            details = _summarize_tool_call(tool_call)
            active.add_event("tool_call", agent=agent_name, **details)

    async def on_tool_result(self, agent_name: str, result: Any) -> None:
        active = self._span_stack[-1] if self._span_stack else self._root_span
        if active:
            output = str(getattr(result, "output", result))
            active.add_event(
                "tool_result",
                agent=agent_name,
                output=truncate_text_by_tokens(
                    output,
                    max_tokens=_TRACE_TEXT_MAX_TOKENS,
                    suffix="",
                ),
            )

    async def on_error(self, agent_name: str, error: Exception) -> None:
        run_span = self._run_span_stack.pop() if self._run_span_stack else None
        if run_span:
            run_span.set_error(error)
            self._finish_until(run_span)
        self._root_span = self._run_span_stack[-1] if self._run_span_stack else None
        self._current_step_span = None
        self._end_execution()

    async def on_run_end(self, agent_name: str, output: str) -> None:
        run_span = self._run_span_stack.pop() if self._run_span_stack else None
        if run_span:
            run_span.attributes["output"] = truncate_text_by_tokens(
                output,
                max_tokens=_TRACE_TEXT_MAX_TOKENS,
                suffix="",
            )
            self._finish_until(run_span)
        self._root_span = self._run_span_stack[-1] if self._run_span_stack else None
        self._current_step_span = None
        self._end_execution()

    # ------------------------------------------------------------------
    # Orchestration-level extensions
    # ------------------------------------------------------------------

    async def on_pipeline_start(self, pipeline_name: str) -> None:
        self._begin_execution()
        span = self._new_span(f"pipeline.{pipeline_name}", kind="internal")
        span.attributes["orchestration"] = "pipeline"
        self._pipeline_span_stack.append(span)
        self._console.print_start(
            kind=span.kind,
            name=span.name,
            indent=max(len(self._span_stack) - 1, 0),
            attributes={"orchestration": "pipeline"},
        )

    async def on_pipeline_end(self, pipeline_name: str) -> None:
        span = self._pipeline_span_stack.pop() if self._pipeline_span_stack else None
        if span:
            self._finish_until(span)
        self._end_execution()

    async def on_dag_start(self, dag_name: str) -> None:
        self._begin_execution()
        span = self._new_span(f"dag.{dag_name}", kind="internal")
        span.attributes["orchestration"] = "dag"
        self._dag_span_stack.append(span)
        self._console.print_start(
            kind=span.kind,
            name=span.name,
            indent=max(len(self._span_stack) - 1, 0),
            attributes={"orchestration": "dag"},
        )

    async def on_dag_end(self, dag_name: str) -> None:
        span = self._dag_span_stack.pop() if self._dag_span_stack else None
        if span:
            self._finish_until(span)
        self._end_execution()

    async def on_dag_node_start(self, node_id: str) -> None:
        span = self._new_span(f"dag_node.{node_id}", kind="internal")
        self._dag_node_span_stack.append(span)
        self._console.print_start(
            kind=span.kind,
            name=span.name,
            indent=max(len(self._span_stack) - 1, 0),
            attributes={"node": node_id},
        )

    async def on_dag_node_end(self, node_id: str) -> None:
        span = self._dag_node_span_stack.pop() if self._dag_node_span_stack else None
        if span:
            self._finish_until(span)

    async def on_team_start(self, team_name: str, mode: str) -> None:
        self._begin_execution()
        span = self._new_span(f"team.{team_name}", kind="internal")
        span.attributes["orchestration"] = "team"
        span.attributes["mode"] = mode
        self._team_span_stack.append(span)
        self._console.print_start(
            kind=span.kind,
            name=span.name,
            indent=max(len(self._span_stack) - 1, 0),
            attributes={
                "orchestration": "team",
                "mode": mode,
            },
        )

    async def on_team_end(self, team_name: str, mode: str) -> None:
        span = self._team_span_stack.pop() if self._team_span_stack else None
        if span:
            self._finish_until(span)
        self._end_execution()


def resolve_hooks(
    hooks: DefaultHooks | None,
    config: HarnessConfig | None,
) -> DefaultHooks:
    cfg = config or HarnessConfig.get()
    tracing_cfg = cfg.tracing
    if not tracing_cfg.enabled:
        return DefaultHooks()
    if hooks is not None:
        return hooks
    cached = cfg.get_runtime_hooks()
    if cached is not None:
        return cached
    created = TracingHooks(trace_dir=tracing_cfg.export_path)
    cfg.set_runtime_hooks(created)
    return created
