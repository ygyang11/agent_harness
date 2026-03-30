"""ProgressHooks — user-facing progress output."""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from agent_harness.hooks.base import DefaultHooks
from agent_harness.utils.theme import COLORS, ICONS

if TYPE_CHECKING:
    from agent_harness.approval.types import ApprovalRequest, ApprovalResult
    from agent_harness.core.message import ToolCall
    from agent_harness.llm.types import StreamDelta


class ProgressHooks(DefaultHooks):
    """User-facing progress output: tool calls, streaming content, errors."""

    _MAX_VISIBLE_TOOLS = 3

    def __init__(self, output: Any = None, color: bool = True) -> None:
        self._output = output or sys.stdout
        self._color = color and hasattr(self._output, "isatty") and self._output.isatty()
        self._streaming = False
        self._tool_call_count = 0
        self._tool_error_count = 0
        self._denied_tool_ids: set[str] = set()

    def _c(self, name: str) -> str:
        return COLORS.get(name, "") if self._color else ""

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

    async def on_approval_request(
        self, agent_name: str, request: ApprovalRequest
    ) -> None:
        if self._streaming:
            self._output.write("\n")
            self._output.flush()
            self._streaming = False

    async def on_approval_result(self, agent_name: str, result: ApprovalResult) -> None:
        from agent_harness.approval.types import ApprovalDecision

        if self._streaming:
            self._output.write("\n")
            self._output.flush()
            self._streaming = False
        if result.decision == ApprovalDecision.DENY:
            self._denied_tool_ids.add(result.tool_call_id)
            red, reset = self._c("red"), self._c("reset")
            denied = ICONS.get("denied", "")
            label = result.tool_name or result.tool_call_id
            reason = f" — {result.reason}" if result.reason else ""
            self._write(f"  {red}{denied} Denied: {label}{reason}{reset}\n")

    async def on_tool_result(self, agent_name: str, result: Any) -> None:
        tool_call_id = getattr(result, "tool_call_id", None)
        if tool_call_id and tool_call_id in self._denied_tool_ids:
            return
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
        self._denied_tool_ids.clear()

    def _write(self, text: str) -> None:
        self._output.write(text)
        self._output.flush()
