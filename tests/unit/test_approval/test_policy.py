"""Tests for ApprovalPolicy."""
from agent_harness.approval.policy import ApprovalPolicy
from agent_harness.approval.types import ApprovalAction
from agent_harness.core.message import ToolCall


class TestApprovalPolicy:
    def test_mode_never_always_executes(self) -> None:
        policy = ApprovalPolicy(mode="never")
        tc = ToolCall(name="dangerous_tool", arguments={})
        assert policy.check(tc) == ApprovalAction.EXECUTE

    def test_mode_never_ignores_deny_list(self) -> None:
        policy = ApprovalPolicy(mode="never", always_deny={"tool_x"})
        tc = ToolCall(name="tool_x", arguments={})
        assert policy.check(tc) == ApprovalAction.EXECUTE

    def test_always_allow_executes(self) -> None:
        policy = ApprovalPolicy(always_allow={"safe_tool"})
        tc = ToolCall(name="safe_tool", arguments={})
        assert policy.check(tc) == ApprovalAction.EXECUTE

    def test_always_deny_denies(self) -> None:
        policy = ApprovalPolicy(always_deny={"banned_tool"})
        tc = ToolCall(name="banned_tool", arguments={})
        assert policy.check(tc) == ApprovalAction.DENY

    def test_deny_overrides_allow(self) -> None:
        policy = ApprovalPolicy(
            always_allow={"tool_x"}, always_deny={"tool_x"},
        )
        tc = ToolCall(name="tool_x", arguments={})
        assert policy.check(tc) == ApprovalAction.DENY

    def test_unknown_tool_asks_in_auto(self) -> None:
        policy = ApprovalPolicy(mode="auto")
        tc = ToolCall(name="unknown_tool", arguments={})
        assert policy.check(tc) == ApprovalAction.ASK

    def test_unknown_tool_asks_in_always(self) -> None:
        policy = ApprovalPolicy(mode="always")
        tc = ToolCall(name="any_tool", arguments={})
        assert policy.check(tc) == ApprovalAction.ASK

    def test_session_allow_remembered(self) -> None:
        policy = ApprovalPolicy(mode="auto")
        policy.grant_session("my_tool")
        tc = ToolCall(name="my_tool", arguments={})
        assert policy.check(tc) == ApprovalAction.EXECUTE

    def test_reset_session_clears_grants(self) -> None:
        policy = ApprovalPolicy(mode="auto")
        policy.grant_session("my_tool")
        policy.reset_session()
        tc = ToolCall(name="my_tool", arguments={})
        assert policy.check(tc) == ApprovalAction.ASK

    def test_session_allow_does_not_override_deny(self) -> None:
        policy = ApprovalPolicy(mode="auto", always_deny={"tool_x"})
        policy.grant_session("tool_x")
        tc = ToolCall(name="tool_x", arguments={})
        assert policy.check(tc) == ApprovalAction.DENY

    def test_default_mode_is_auto(self) -> None:
        policy = ApprovalPolicy()
        tc = ToolCall(name="any_tool", arguments={})
        assert policy.check(tc) == ApprovalAction.ASK
