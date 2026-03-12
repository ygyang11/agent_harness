"""Regression tests for Tracer contextvars fix and RateLimiter lock fix."""
from __future__ import annotations

import asyncio
import pytest

from agent_harness.tracing.tracer import Tracer, TraceCollector


class TestTracerConcurrency:
    """Issue #4: Concurrent spans must get correct trace/parent IDs."""

    @pytest.mark.asyncio
    async def test_concurrent_spans_get_independent_traces(self) -> None:
        """Two concurrent root spans must not share trace_id."""
        collector = TraceCollector()
        tracer = Tracer(collector=collector)

        trace_ids: list[str] = []

        async def worker(name: str) -> None:
            async with tracer.span(f"root.{name}", kind="agent") as s:
                trace_ids.append(s.trace_id)
                await asyncio.sleep(0.01)  # simulate work

        await asyncio.gather(worker("a"), worker("b"))

        # Each worker should get its own trace_id
        assert len(trace_ids) == 2
        assert trace_ids[0] != trace_ids[1]

    @pytest.mark.asyncio
    async def test_nested_spans_share_trace_id(self) -> None:
        """Child spans within the same async flow share the parent's trace_id."""
        collector = TraceCollector()
        tracer = Tracer(collector=collector)

        async with tracer.span("root", kind="agent") as root:
            async with tracer.span("child", kind="internal") as child:
                assert child.trace_id == root.trace_id
                assert child.parent_span_id == root.span_id

    @pytest.mark.asyncio
    async def test_concurrent_nested_spans_correct_parents(self) -> None:
        """Concurrent tasks with nested spans must have correct parent chains."""
        collector = TraceCollector()
        tracer = Tracer(collector=collector)

        results: dict[str, dict] = {}

        async def worker(name: str) -> None:
            async with tracer.span(f"root.{name}") as root:
                await asyncio.sleep(0.01)
                async with tracer.span(f"child.{name}") as child:
                    results[name] = {
                        "root_trace": root.trace_id,
                        "child_trace": child.trace_id,
                        "child_parent": child.parent_span_id,
                        "root_span": root.span_id,
                    }

        await asyncio.gather(worker("a"), worker("b"))

        # Each worker's child should point to its own root, not the other's
        for name in ("a", "b"):
            assert results[name]["child_trace"] == results[name]["root_trace"]
            assert results[name]["child_parent"] == results[name]["root_span"]

        # Different workers should have different trace_ids
        assert results["a"]["root_trace"] != results["b"]["root_trace"]
