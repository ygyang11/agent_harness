"""Tracer and Span model for agent_harness observability.

Provides structured tracing inspired by OpenTelemetry, but lightweight
and purpose-built for agent execution tracking.
"""
from __future__ import annotations

import uuid
import logging
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Any, AsyncIterator, Callable, Iterator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SpanEvent(BaseModel):
    """An event that occurred during a span."""
    name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    attributes: dict[str, Any] = Field(default_factory=dict)


class Span(BaseModel):
    """A single unit of work in a trace.

    Spans form a tree structure: each span can have a parent and children.
    """
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: str | None = None
    name: str
    kind: str = "internal"  # "agent", "llm", "tool", "memory", "internal"
    start_time: datetime = Field(default_factory=lambda: datetime.now())
    end_time: datetime | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[SpanEvent] = Field(default_factory=list)
    status: str = "ok"  # "ok", "error"
    error_message: str | None = None

    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000

    def add_event(self, name: str, **attributes: Any) -> None:
        self.events.append(SpanEvent(name=name, attributes=attributes))

    def set_error(self, error: Exception) -> None:
        self.status = "error"
        self.error_message = str(error)

    def finish(self) -> None:
        self.end_time = datetime.now()


class TraceCollector:
    """Collects finished spans for export."""

    def __init__(self) -> None:
        self._spans: list[Span] = []

    def add_span(self, span: Span) -> None:
        self._spans.append(span)

    def get_spans(self) -> list[Span]:
        return list(self._spans)

    def get_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self._spans if s.trace_id == trace_id]

    def clear(self) -> None:
        self._spans.clear()


class Tracer:
    """Creates and manages spans for tracing agent execution.

    Example:
        tracer = Tracer()
        async with tracer.span("agent.run", kind="agent") as s:
            s.attributes["input"] = user_input
            result = await do_work()
            s.attributes["output"] = result
    """

    def __init__(self, collector: TraceCollector | None = None, enabled: bool = True) -> None:
        self._collector = collector or TraceCollector()
        self._enabled = enabled
        self._current_trace_id: str | None = None
        self._current_span_id: str | None = None

    @property
    def collector(self) -> TraceCollector:
        return self._collector

    @asynccontextmanager
    async def span(self, name: str, kind: str = "internal", **attributes: Any) -> AsyncIterator[Span]:
        """Create a span as an async context manager."""
        if not self._enabled:
            yield Span(name=name, kind=kind)
            return

        trace_id = self._current_trace_id or uuid.uuid4().hex
        parent_span_id = self._current_span_id

        s = Span(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            attributes=attributes,
        )

        prev_trace_id = self._current_trace_id
        prev_span_id = self._current_span_id
        self._current_trace_id = trace_id
        self._current_span_id = s.span_id

        try:
            yield s
        except Exception as e:
            s.set_error(e)
            raise
        finally:
            s.finish()
            self._collector.add_span(s)
            self._current_trace_id = prev_trace_id
            self._current_span_id = prev_span_id

    def trace(self, name: str, kind: str = "internal") -> Callable:
        """Decorator to trace an async function."""
        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                async with self.span(name, kind=kind) as s:
                    s.attributes["args_count"] = len(args)
                    result = await fn(*args, **kwargs)
                    return result
            return wrapper
        return decorator
