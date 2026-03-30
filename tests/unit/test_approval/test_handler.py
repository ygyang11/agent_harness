"""Tests for approval handlers."""
from __future__ import annotations

import io
from unittest.mock import patch

from agent_harness.approval.handler import StdinApprovalHandler
from agent_harness.approval.types import ApprovalDecision, ApprovalRequest
from agent_harness.core.message import ToolCall


class TestStdinApprovalHandler:
    async def test_default_is_allow_once(self) -> None:
        output = io.StringIO()
        handler = StdinApprovalHandler(output=output, color=False)
        tc = ToolCall(name="my_tool", arguments={"x": "1"})
        request = ApprovalRequest(tool_call=tc, agent_name="agent")

        with patch("builtins.input", return_value=""):
            result = await handler.request_approval(request)

        assert result.decision == ApprovalDecision.ALLOW_ONCE
        assert result.tool_call_id == tc.id

    async def test_yes_is_allow_once(self) -> None:
        handler = StdinApprovalHandler(output=io.StringIO(), color=False)
        tc = ToolCall(name="my_tool", arguments={})
        request = ApprovalRequest(tool_call=tc, agent_name="agent")

        with patch("builtins.input", return_value="y"):
            result = await handler.request_approval(request)
        assert result.decision == ApprovalDecision.ALLOW_ONCE

    async def test_always_is_allow_session(self) -> None:
        handler = StdinApprovalHandler(output=io.StringIO(), color=False)
        tc = ToolCall(name="my_tool", arguments={})
        request = ApprovalRequest(tool_call=tc, agent_name="agent")

        with patch("builtins.input", return_value="a"):
            result = await handler.request_approval(request)
        assert result.decision == ApprovalDecision.ALLOW_SESSION

    async def test_no_is_deny(self) -> None:
        handler = StdinApprovalHandler(output=io.StringIO(), color=False)
        tc = ToolCall(name="my_tool", arguments={})
        request = ApprovalRequest(tool_call=tc, agent_name="agent")

        with patch("builtins.input", return_value="n"):
            result = await handler.request_approval(request)
        assert result.decision == ApprovalDecision.DENY

    async def test_output_shows_tool_info(self) -> None:
        output = io.StringIO()
        handler = StdinApprovalHandler(output=output, color=False)
        tc = ToolCall(name="web_search", arguments={"query": "test"})
        request = ApprovalRequest(tool_call=tc, agent_name="agent")

        with patch("builtins.input", return_value="y"):
            await handler.request_approval(request)

        written = output.getvalue()
        assert "web_search" in written
        assert "query" in written
        assert "Allow?" in written
