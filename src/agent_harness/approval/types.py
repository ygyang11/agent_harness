"""Approval data types for human-in-the-loop tool execution."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from agent_harness.core.message import ToolCall


class ApprovalDecision(StrEnum):
    """User's response to an approval request."""

    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"


class ApprovalAction(StrEnum):
    """Result of policy check — what to do with a tool call."""

    EXECUTE = "execute"
    ASK = "ask"
    DENY = "deny"


class ApprovalRequest(BaseModel):
    """Information presented to the user for approval."""

    tool_call: ToolCall
    agent_name: str
    step: int | None = None


class ApprovalResult(BaseModel):
    """User's decision on an approval request."""

    tool_call_id: str
    tool_name: str | None = None
    decision: ApprovalDecision
    reason: str | None = None
