"""Tests for approval data types."""
from agent_harness.approval.types import (
    ApprovalAction,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResult,
)
from agent_harness.core.message import ToolCall


class TestApprovalDecision:
    def test_values(self) -> None:
        assert ApprovalDecision.ALLOW_ONCE == "allow_once"
        assert ApprovalDecision.ALLOW_SESSION == "allow_session"
        assert ApprovalDecision.DENY == "deny"

    def test_is_str_enum(self) -> None:
        assert isinstance(ApprovalDecision.DENY, str)


class TestApprovalAction:
    def test_values(self) -> None:
        assert ApprovalAction.EXECUTE == "execute"
        assert ApprovalAction.ASK == "ask"
        assert ApprovalAction.DENY == "deny"


class TestApprovalRequest:
    def test_creation(self) -> None:
        tc = ToolCall(name="my_tool", arguments={"x": 1})
        req = ApprovalRequest(tool_call=tc, agent_name="agent1")
        assert req.tool_call.name == "my_tool"
        assert req.agent_name == "agent1"
        assert req.step is None

    def test_with_step(self) -> None:
        tc = ToolCall(name="my_tool", arguments={})
        req = ApprovalRequest(tool_call=tc, agent_name="agent1", step=3)
        assert req.step == 3


class TestApprovalResult:
    def test_creation(self) -> None:
        result = ApprovalResult(
            tool_call_id="call_123",
            decision=ApprovalDecision.ALLOW_ONCE,
        )
        assert result.tool_call_id == "call_123"
        assert result.decision == ApprovalDecision.ALLOW_ONCE
        assert result.reason is None

    def test_with_reason(self) -> None:
        result = ApprovalResult(
            tool_call_id="call_456",
            decision=ApprovalDecision.DENY,
            reason="Too dangerous",
        )
        assert result.reason == "Too dangerous"
