"""Tests for ConsoleExporter rendering helpers."""
from __future__ import annotations

from io import StringIO

from agent_harness.tracing.exporters.console import ConsoleExporter
from agent_harness.tracing.tracer import Span
from agent_harness.utils.token_counter import count_tokens


def test_print_start_renders_marker_and_attributes() -> None:
    stream = StringIO()
    exporter = ConsoleExporter(stream=stream, color=False)

    exporter.print_start(
        kind="agent",
        name="agent.researcher",
        indent=1,
        attributes={"input": "hello world"},
    )

    output = stream.getvalue()
    assert "  ▶ [agent] agent.researcher (start)\n" in output
    assert "    input: hello world\n" in output


def test_print_start_truncates_attribute_by_tokens() -> None:
    stream = StringIO()
    exporter = ConsoleExporter(stream=stream, color=False)
    long_text = "signal " * 500

    exporter.print_start(
        kind="agent",
        name="agent.researcher",
        attributes={"input": long_text},
    )

    line = next(line for line in stream.getvalue().splitlines() if "input:" in line)
    value = line.split("input: ", 1)[1]
    assert count_tokens(value) <= 40


def test_export_one_respects_explicit_indent() -> None:
    stream = StringIO()
    exporter = ConsoleExporter(stream=stream, color=False)
    span = Span(name="agent.sample", kind="agent")
    span.finish()

    exporter.export_one(span, indent=2)

    output = stream.getvalue()
    assert output.startswith("    ✓ [agent] agent.sample (")


def test_event_attributes_render_as_key_value_pairs() -> None:
    stream = StringIO()
    exporter = ConsoleExporter(stream=stream, color=False)
    span = Span(name="step.1", kind="internal")
    span.add_event(
        "tool_call",
        agent="researcher.executor",
        tool="web_search",
        args={"query": "solar market size"},
    )
    span.finish()

    exporter.export_one(span, indent=0)

    output = stream.getvalue()
    assert "• tool_call {" in output
    assert "tool=web_search" in output
    assert "args={'query': 'solar market size'}" in output
