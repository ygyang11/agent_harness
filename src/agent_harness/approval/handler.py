"""Approval handlers — abstract interface and stdin default implementation."""
from __future__ import annotations

import asyncio
import sys
from abc import ABC, abstractmethod
from typing import Any

from agent_harness.approval.types import ApprovalDecision, ApprovalRequest, ApprovalResult
from agent_harness.utils.theme import COLORS, ICONS


class ApprovalHandler(ABC):
    """Abstract interface for getting approval decisions from a human."""

    @abstractmethod
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResult: ...


class StdinApprovalHandler(ApprovalHandler):
    """Interactive stdin-based approval handler.

    Displays the tool call and reads user input:
      [Y]es / [A]lways / [N]o (default: Y)
    """

    def __init__(self, output: Any = None, color: bool = True) -> None:
        self._output = output or sys.stdout
        self._color = color and hasattr(self._output, "isatty") and self._output.isatty()

    def _c(self, name: str) -> str:
        return COLORS.get(name, "") if self._color else ""

    def _icon(self, name: str) -> str:
        return ICONS.get(name, "")

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        tc = request.tool_call
        bold, yellow, reset = self._c("bold"), self._c("yellow"), self._c("reset")
        marker, lock = self._icon("marker"), self._icon("approval")
        args_preview = ", ".join(f'{k}="{v}"' for k, v in tc.arguments.items())
        self._output.write(
            f"{yellow}{marker}{reset} {lock} {bold}{tc.name}{reset}({args_preview})\n"
        )
        self._output.write("  Allow? [Y]es / [A]lways / [N]o <reason> (default: Y): ")
        self._output.flush()

        loop = asyncio.get_running_loop()
        raw: str = await loop.run_in_executor(None, input)
        choice = raw.strip().lower()

        reason: str | None = None
        if choice.startswith("n"):
            decision = ApprovalDecision.DENY
            # Extract reason after "n" or "no": "n too dangerous" → "too dangerous"
            rest = choice.split(None, 1)
            if len(rest) > 1:
                reason = rest[1]
        elif choice in ("a", "always"):
            decision = ApprovalDecision.ALLOW_SESSION
        else:
            decision = ApprovalDecision.ALLOW_ONCE

        return ApprovalResult(
            tool_call_id=tc.id, tool_name=tc.name, decision=decision, reason=reason,
        )
