"""Approval policy — determines which tool calls require human approval."""
from __future__ import annotations

from agent_harness.approval.types import ApprovalAction
from agent_harness.core.message import ToolCall


class ApprovalPolicy:
    """Determines which tool calls require human approval.

    Args:
        mode: "always" — ask for every tool call (unless overridden by allow/deny sets).
              "auto" — only ask for tools not in always_allow set (default).
              "never" — skip all approvals.
        always_allow: Tool names that never require approval.
        always_deny: Tool names that are always rejected (auto-denied without asking).
    """

    def __init__(
        self,
        *,
        mode: str = "auto",
        always_allow: set[str] | None = None,
        always_deny: set[str] | None = None,
    ) -> None:
        self._mode = mode
        self._always_allow = always_allow or set()
        self._always_deny = always_deny or set()
        self._session_allow: set[str] = set()

    def check(self, tool_call: ToolCall) -> ApprovalAction:
        """Determine what to do with a tool call.

        Priority: always_deny > always_allow > session_allow > ask user.
        """
        if self._mode == "never":
            return ApprovalAction.EXECUTE
        if tool_call.name in self._always_deny:
            return ApprovalAction.DENY
        if tool_call.name in self._always_allow:
            return ApprovalAction.EXECUTE
        if tool_call.name in self._session_allow:
            return ApprovalAction.EXECUTE
        return ApprovalAction.ASK

    def grant_session(self, tool_name: str) -> None:
        """Grant session-level approval for a tool (ALLOW_SESSION)."""
        self._session_allow.add(tool_name)

    def reset_session(self) -> None:
        """Clear session-level approvals."""
        self._session_allow.clear()
