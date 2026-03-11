"""Console exporter for trace spans.

Prints spans to the terminal with indentation and optional color.
Useful for development and debugging.
"""
from __future__ import annotations

from typing import TextIO
import sys

from agent_harness.tracing.tracer import Span

# ANSI color codes
_COLORS = {
    "agent": "\033[1;34m",   # Bold Blue
    "llm": "\033[1;32m",     # Bold Green
    "tool": "\033[1;33m",    # Bold Yellow
    "memory": "\033[1;35m",  # Bold Magenta
    "internal": "\033[0;37m", # Gray
    "error": "\033[1;31m",   # Bold Red
    "reset": "\033[0m",
}


class ConsoleExporter:
    """Export spans to the console with colored, indented output."""

    def __init__(self, stream: TextIO = sys.stderr, color: bool = True) -> None:
        self._stream = stream
        self._color = color

    def export(self, spans: list[Span]) -> None:
        """Export a list of spans to the console."""
        # Group by trace_id and build tree
        for span in spans:
            self._print_span(span)

    def _print_span(self, span: Span, indent: int = 0) -> None:
        prefix = "  " * indent
        kind_color = _COLORS.get(span.kind, _COLORS["internal"]) if self._color else ""
        reset = _COLORS["reset"] if self._color else ""
        error_color = _COLORS["error"] if self._color else ""

        duration = f"{span.duration_ms:.1f}ms" if span.duration_ms is not None else "..."
        status_indicator = "✓" if span.status == "ok" else f"{error_color}✗{reset}"

        line = f"{prefix}{status_indicator} {kind_color}[{span.kind}]{reset} {span.name} ({duration})"
        self._stream.write(line + "\n")

        if span.attributes:
            for key, value in span.attributes.items():
                val_str = str(value)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                self._stream.write(f"{prefix}  {key}: {val_str}\n")

        if span.error_message:
            self._stream.write(f"{prefix}  {error_color}error: {span.error_message}{reset}\n")

        for event in span.events:
            self._stream.write(f"{prefix}  • {event.name}")
            if event.attributes:
                self._stream.write(f" {event.attributes}")
            self._stream.write("\n")
