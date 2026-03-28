"""Agent lifecycle hooks for extensible behavior."""
from __future__ import annotations

import contextvars
import sys
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agent_harness.core.config import HarnessConfig
from agent_harness.tracing.exporters.console import ConsoleExporter
from agent_harness.tracing.exporters.json_file import JsonFileExporter
from agent_harness.tracing.tracer import Span
from agent_harness.utils.token_counter import truncate_text_by_tokens

if TYPE_CHECKING:
    from agent_harness.core.message import ToolCall
    from agent_harness.llm.types import StreamDelta

_active_orchestration_parent: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "_active_orchestration_parent", default=None
)
_active_step_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "_active_step_span", default=None
)
_active_run_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "_active_run_span", default=None
)
_streaming_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_streaming_active", default=False
)


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

    async def on_llm_stream_delta(self, agent_name: str, delta: StreamDelta) -> None:
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

    async def on_pipeline_start(self, pipeline_name: str) -> None:
        pass

    async def on_pipeline_end(self, pipeline_name: str) -> None:
        pass

    async def on_dag_start(self, dag_name: str) -> None:
        pass

    async def on_dag_end(self, dag_name: str) -> None:
        pass

    async def on_dag_node_start(self, node_id: str) -> None:
        pass

    async def on_dag_node_end(self, node_id: str) -> None:
        pass

    async def on_team_start(self, team_name: str, mode: str) -> None:
        pass

    async def on_team_end(self, team_name: str, mode: str) -> None:
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
        exporter: str = "both",
        console_color: bool = True,
    ) -> None:
        self._trace_dir = trace_dir
        self._console_enabled = exporter in ("console", "both")
        self._json_enabled = exporter in ("json_file", "both")
        self._console = (
            ConsoleExporter(stream=sys.stderr, color=console_color)
            if self._console_enabled else None
        )

        # Span tracking
        self._trace_id: str = ""
        self._root_span: Span | None = None
        self._span_stack: list[Span] = []
        self._run_span_stack: list[Span] = []
        self._pipeline_span_stack: list[Span] = []
        self._dag_span_stack: list[Span] = []
        self._team_span_stack: list[Span] = []
        self._dag_node_span_stack: list[Span] = []
        self._all_spans: list[Span] = []
        self._span_depth_map: dict[str, int] = {}
        self._depth = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cwrite(self, text: str) -> None:
        if self._console is not None:
            self._console.write_inline(text)

    def _cprint_start(self, **kwargs: Any) -> None:
        if self._console is not None:
            self._console.print_start(**kwargs)

    def _cexport_one(self, span: Span, *, indent: int) -> None:
        if self._console is not None:
            self._console.export_one(span, indent=indent)

    def _new_span(
        self,
        name: str,
        kind: str = "internal",
        *,
        parent_span: Span | None = None,
        **attrs: Any,
    ) -> Span:
        parent_id: str | None
        if parent_span is not None:
            parent_id = parent_span.span_id
        else:
            parent_id = self._span_stack[-1].span_id if self._span_stack else None

        span = Span(
            trace_id=self._trace_id,
            parent_span_id=parent_id,
            name=name,
            kind=kind,
            attributes=attrs,
        )

        if parent_id and parent_id in self._span_depth_map:
            self._span_depth_map[span.span_id] = self._span_depth_map[parent_id] + 1
        else:
            self._span_depth_map[span.span_id] = 0

        self._span_stack.append(span)
        return span

    def _finish_span(self, span: Span) -> None:
        indent = self._span_depth_map.get(span.span_id, 0)
        span.finish()
        self._all_spans.append(span)
        if span in self._span_stack:
            self._span_stack.remove(span)
        self._cexport_one(span, indent=indent)

    def _export_all(self) -> None:
        """Flush all collected spans to JSON file."""
        if not self._json_enabled or not self._all_spans:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exporter = JsonFileExporter(path=f"{self._trace_dir}/{ts}.jsonl")
        exporter.export(self._all_spans)

    def _finish_until(self, target: Span) -> None:
        if target not in self._span_stack:
            return
        self._finish_span(target)

    def _begin_execution(self) -> None:
        self._depth += 1
        if self._depth == 1:
            self._trace_id = uuid.uuid4().hex
            self._root_span = None
            self._span_stack = []
            self._run_span_stack = []
            self._pipeline_span_stack = []
            self._dag_span_stack = []
            self._team_span_stack = []
            self._dag_node_span_stack = []
            self._all_spans = []
            self._span_depth_map = {}

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
        orchestration_parent = _active_orchestration_parent.get(None)
        run_span = self._new_span(
            f"agent.{agent_name}",
            kind="agent",
            parent_span=orchestration_parent,
            input=truncated_input,
        )
        self._run_span_stack.append(run_span)
        _active_run_span.set(run_span)
        self._root_span = run_span
        self._cprint_start(
            kind=run_span.kind,
            name=run_span.name,
            indent=self._span_depth_map.get(run_span.span_id, 0),
            attributes={"input": truncated_input},
        )

    async def on_step_start(self, agent_name: str, step: int) -> None:
        run_span = _active_run_span.get(None)
        step_span = self._new_span(
            f"step.{step}", kind="internal", parent_span=run_span, agent=agent_name,
        )
        _active_step_span.set(step_span)
        indent = self._span_depth_map.get(step_span.span_id, 0)
        self._cwrite(f"{'  ' * indent}⟳ step.{step}\n")

    async def on_step_end(self, agent_name: str, step: int) -> None:
        if _streaming_active.get(False):
            self._cwrite("\n")
            _streaming_active.set(False)
        step_span = _active_step_span.get(None)
        if step_span:
            step_span.finish()
            self._all_spans.append(step_span)
            if step_span in self._span_stack:
                self._span_stack.remove(step_span)
            indent = self._span_depth_map.get(step_span.span_id, 0)
            duration = (
                f"{step_span.duration_ms:.1f}ms"
                if step_span.duration_ms is not None else "..."
            )
            self._cwrite(f"{'  ' * indent}✓ {step_span.name} ({duration})\n")
            _active_step_span.set(None)

    async def on_llm_call(self, agent_name: str, messages: list[Any]) -> None:
        active = _active_step_span.get(None) or _active_run_span.get(None)
        if active:
            active.add_event("llm_call", agent=agent_name, message_count=len(messages))
            indent = self._span_depth_map.get(active.span_id, 0) + 1
            self._cwrite(
                f"{'  ' * indent}• llm_call "
                f"{{agent={agent_name}, message_count={len(messages)}}}\n"
            )

    async def on_llm_stream_delta(self, agent_name: str, delta: StreamDelta) -> None:
        if delta.chunk.delta_content:
            if not _streaming_active.get(False):
                _streaming_active.set(True)
                step_span = _active_step_span.get(None)
                depth = (
                    self._span_depth_map.get(step_span.span_id, 0) + 1
                    if step_span else 0
                )
                self._cwrite("  " * depth + "▸ ")
            self._cwrite(delta.chunk.delta_content)

    async def on_tool_call(self, agent_name: str, tool_call: ToolCall) -> None:
        if _streaming_active.get(False):
            self._cwrite("\n")
            _streaming_active.set(False)
        active = _active_step_span.get(None) or _active_run_span.get(None)
        if active:
            details = _summarize_tool_call(tool_call)
            active.add_event("tool_call", agent=agent_name, **details)
            indent = self._span_depth_map.get(active.span_id, 0) + 1
            self._cwrite(
                f"{'  ' * indent}• tool_call "
                f"{{agent={agent_name}, tool={details['tool']}, args={details['args']}}}\n"
            )

    async def on_tool_result(self, agent_name: str, result: Any) -> None:
        active = _active_step_span.get(None) or _active_run_span.get(None)
        if active:
            output = result.content if hasattr(result, "content") else str(result)
            truncated = truncate_text_by_tokens(
                output, max_tokens=_TRACE_TEXT_MAX_TOKENS, suffix="",
            )
            active.add_event("tool_result", agent=agent_name, output=truncated)
            indent = self._span_depth_map.get(active.span_id, 0) + 1
            self._cwrite(
                f"{'  ' * indent}• tool_result {{agent={agent_name}, output={truncated}}}\n"
            )

    async def on_error(self, agent_name: str, error: Exception) -> None:
        if _streaming_active.get(False):
            self._cwrite("\n")
            _streaming_active.set(False)
        step_span = _active_step_span.get(None)
        if step_span:
            step_span.finish()
            self._all_spans.append(step_span)
            if step_span in self._span_stack:
                self._span_stack.remove(step_span)
            indent = self._span_depth_map.get(step_span.span_id, 0)
            duration = (
                f"{step_span.duration_ms:.1f}ms"
                if step_span.duration_ms is not None else "..."
            )
            self._cwrite(f"{'  ' * indent}✗ {step_span.name} ({duration})\n")
            _active_step_span.set(None)
        run_span = _active_run_span.get(None)
        _active_run_span.set(None)
        if run_span:
            run_span.set_error(error)
            self._finish_span(run_span)
            if run_span in self._run_span_stack:
                self._run_span_stack.remove(run_span)
        self._root_span = self._run_span_stack[-1] if self._run_span_stack else None
        self._end_execution()

    async def on_run_end(self, agent_name: str, output: str) -> None:
        run_span = _active_run_span.get(None)
        _active_run_span.set(None)
        if run_span:
            run_span.attributes["output"] = truncate_text_by_tokens(
                output,
                max_tokens=_TRACE_TEXT_MAX_TOKENS,
                suffix="",
            )
            self._finish_span(run_span)
            if run_span in self._run_span_stack:
                self._run_span_stack.remove(run_span)
        self._root_span = self._run_span_stack[-1] if self._run_span_stack else None
        self._end_execution()

    # ------------------------------------------------------------------
    # Orchestration-level extensions
    # ------------------------------------------------------------------

    async def on_pipeline_start(self, pipeline_name: str) -> None:
        self._begin_execution()
        span = self._new_span(f"pipeline.{pipeline_name}", kind="orchestration")
        span.attributes["orchestration"] = "pipeline"
        self._pipeline_span_stack.append(span)
        _active_orchestration_parent.set(span)
        self._cprint_start(
            kind=span.kind,
            name=span.name,
            indent=self._span_depth_map.get(span.span_id, 0),
            attributes={"orchestration": "pipeline"},
        )

    async def on_pipeline_end(self, pipeline_name: str) -> None:
        _active_orchestration_parent.set(None)
        span = self._pipeline_span_stack.pop() if self._pipeline_span_stack else None
        if span:
            self._finish_until(span)
        self._end_execution()

    async def on_dag_start(self, dag_name: str) -> None:
        self._begin_execution()
        span = self._new_span(f"dag.{dag_name}", kind="orchestration")
        span.attributes["orchestration"] = "dag"
        self._dag_span_stack.append(span)
        self._cprint_start(
            kind=span.kind,
            name=span.name,
            indent=self._span_depth_map.get(span.span_id, 0),
            attributes={"orchestration": "dag"},
        )

    async def on_dag_end(self, dag_name: str) -> None:
        span = self._dag_span_stack.pop() if self._dag_span_stack else None
        if span:
            self._finish_until(span)
        self._end_execution()

    async def on_dag_node_start(self, node_id: str) -> None:
        dag_span = self._dag_span_stack[-1] if self._dag_span_stack else None
        span = self._new_span(
            f"dag_node.{node_id}",
            kind="orchestration",
            parent_span=dag_span,
            node=node_id,
        )
        self._dag_node_span_stack.append(span)
        _active_orchestration_parent.set(span)
        self._cprint_start(
            kind=span.kind,
            name=span.name,
            indent=self._span_depth_map.get(span.span_id, 0),
            attributes={"node": node_id},
        )

    async def on_dag_node_end(self, node_id: str) -> None:
        _active_orchestration_parent.set(None)
        span = self._dag_node_span_stack.pop() if self._dag_node_span_stack else None
        if span:
            self._finish_span(span)

    async def on_team_start(self, team_name: str, mode: str) -> None:
        self._begin_execution()
        span = self._new_span(f"team.{team_name}", kind="orchestration")
        span.attributes["orchestration"] = "team"
        span.attributes["mode"] = mode
        self._team_span_stack.append(span)
        _active_orchestration_parent.set(span)
        self._cprint_start(
            kind=span.kind,
            name=span.name,
            indent=self._span_depth_map.get(span.span_id, 0),
            attributes={
                "orchestration": "team",
                "mode": mode,
            },
        )

    async def on_team_end(self, team_name: str, mode: str) -> None:
        _active_orchestration_parent.set(None)
        span = self._team_span_stack.pop() if self._team_span_stack else None
        if span:
            self._finish_until(span)
        self._end_execution()


_PROGRESS_COLORS = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "reset": "\033[0m",
}


class ProgressHooks(DefaultHooks):
    """User-facing progress output: tool calls, streaming content, errors."""

    _MAX_VISIBLE_TOOLS = 3

    def __init__(self, output: Any = None, color: bool = True) -> None:
        self._output = output or sys.stdout
        self._color = color and hasattr(self._output, "isatty") and self._output.isatty()
        self._streaming = False
        self._tool_call_count = 0
        self._tool_error_count = 0

    def _c(self, name: str) -> str:
        return _PROGRESS_COLORS.get(name, "") if self._color else ""

    async def on_tool_call(self, agent_name: str, tool_call: ToolCall) -> None:
        if self._streaming:
            self._output.write("\n")
            self._output.flush()
            self._streaming = False
        self._tool_call_count += 1
        if self._tool_call_count <= self._MAX_VISIBLE_TOOLS:
            bold, reset = self._c("bold"), self._c("reset")
            args_preview = ", ".join(
                f'{k}="{v}"' for k, v in tool_call.arguments.items()
            )
            yellow, reset2 = self._c("yellow"), self._c("reset")
            prefix = f"{yellow}⏺{reset2} " if self._tool_call_count == 1 else "  "
            self._write(f"{prefix}⚡ {bold}{tool_call.name}{reset}({args_preview})\n")

    async def on_tool_result(self, agent_name: str, result: Any) -> None:
        if getattr(result, "is_error", False):
            self._tool_error_count += 1

    async def on_llm_stream_delta(self, agent_name: str, delta: StreamDelta) -> None:
        if delta.chunk.delta_content:
            if not self._streaming:
                self._streaming = True
                self._write("⏺ ")
            self._output.write(delta.chunk.delta_content)
            self._output.flush()

    async def on_step_end(self, agent_name: str, step: int) -> None:
        if self._streaming:
            self._output.write("\n")
            self._output.flush()
            self._streaming = False
        if self._tool_call_count > 0:
            self._print_tool_summary()

    async def on_error(self, agent_name: str, error: Exception) -> None:
        if self._streaming:
            self._output.write("\n")
            self._output.flush()
            self._streaming = False
        if self._tool_call_count > 0:
            self._print_tool_summary()
        red, reset = self._c("red"), self._c("reset")
        self._write(f"  {red}❌ Error: {error}{reset}\n")

    def _print_tool_summary(self) -> None:
        green, red, reset = self._c("green"), self._c("red"), self._c("reset")
        total = self._tool_call_count
        if total > self._MAX_VISIBLE_TOOLS:
            overflow = total - self._MAX_VISIBLE_TOOLS
            self._write(f"  ... and {overflow} more tools\n")
        errors = self._tool_error_count
        if errors:
            self._write(
                f"  ⎿ {green}✓ {total - errors}/{total} completed{reset}, "
                f"{red}✗ {errors} failed{reset}\n"
            )
        else:
            self._write(f"  ⎿ {green}✓ {total}/{total} completed{reset}\n")
        self._tool_call_count = 0
        self._tool_error_count = 0

    def _write(self, text: str) -> None:
        self._output.write(text)
        self._output.flush()


class CompositeHooks(DefaultHooks):
    """Runs multiple hooks implementations in sequence."""

    def __init__(self, *hooks: DefaultHooks) -> None:
        self._hooks = [h for h in hooks if h is not None]

    async def on_run_start(self, agent_name: str, input_text: str) -> None:
        for h in self._hooks:
            await h.on_run_start(agent_name, input_text)

    async def on_step_start(self, agent_name: str, step: int) -> None:
        for h in self._hooks:
            await h.on_step_start(agent_name, step)

    async def on_llm_call(self, agent_name: str, messages: list[Any]) -> None:
        for h in self._hooks:
            await h.on_llm_call(agent_name, messages)

    async def on_llm_stream_delta(self, agent_name: str, delta: StreamDelta) -> None:
        for h in self._hooks:
            await h.on_llm_stream_delta(agent_name, delta)

    async def on_tool_call(self, agent_name: str, tool_call: ToolCall) -> None:
        for h in self._hooks:
            await h.on_tool_call(agent_name, tool_call)

    async def on_tool_result(self, agent_name: str, result: Any) -> None:
        for h in self._hooks:
            await h.on_tool_result(agent_name, result)

    async def on_step_end(self, agent_name: str, step: int) -> None:
        for h in self._hooks:
            await h.on_step_end(agent_name, step)

    async def on_run_end(self, agent_name: str, output: str) -> None:
        for h in self._hooks:
            await h.on_run_end(agent_name, output)

    async def on_error(self, agent_name: str, error: Exception) -> None:
        for h in self._hooks:
            await h.on_error(agent_name, error)

    async def on_pipeline_start(self, pipeline_name: str) -> None:
        for h in self._hooks:
            await h.on_pipeline_start(pipeline_name)

    async def on_pipeline_end(self, pipeline_name: str) -> None:
        for h in self._hooks:
            await h.on_pipeline_end(pipeline_name)

    async def on_dag_start(self, dag_name: str) -> None:
        for h in self._hooks:
            await h.on_dag_start(dag_name)

    async def on_dag_end(self, dag_name: str) -> None:
        for h in self._hooks:
            await h.on_dag_end(dag_name)

    async def on_dag_node_start(self, node_id: str) -> None:
        for h in self._hooks:
            await h.on_dag_node_start(node_id)

    async def on_dag_node_end(self, node_id: str) -> None:
        for h in self._hooks:
            await h.on_dag_node_end(node_id)

    async def on_team_start(self, team_name: str, mode: str) -> None:
        for h in self._hooks:
            await h.on_team_start(team_name, mode)

    async def on_team_end(self, team_name: str, mode: str) -> None:
        for h in self._hooks:
            await h.on_team_end(team_name, mode)


def resolve_hooks(
    hooks: DefaultHooks | None,
    config: HarnessConfig | None,
) -> DefaultHooks:
    cfg = config or HarnessConfig.get()

    if hooks is not None:
        return hooks

    if cfg.tracing.enabled:
        cached = cfg.get_runtime_hooks()
        if cached is not None:
            return cached
        created = TracingHooks(
            trace_dir=cfg.tracing.export_path,
            exporter=cfg.tracing.exporter,
        )
        cfg.set_runtime_hooks(created)
        return created

    return ProgressHooks()
