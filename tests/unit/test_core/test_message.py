"""Tests for agent_harness.core.message — Message factories, ToolCall, properties."""
from __future__ import annotations

import pytest

from agent_harness.core.message import Message, MessageChunk, Role, ToolCall, ToolResult


class TestRole:
    @pytest.mark.parametrize(
        ("role", "expected"),
        [
            (Role.SYSTEM, "system"),
            (Role.USER, "user"),
            (Role.ASSISTANT, "assistant"),
            (Role.TOOL, "tool"),
        ],
    )
    def test_role_contract(self, role: Role, expected: str) -> None:
        assert role.value == expected
        assert role == expected
        assert isinstance(role, str)


class TestToolCall:
    def test_auto_id_generation(self) -> None:
        tc = ToolCall(name="search", arguments={"q": "hello"})
        assert tc.id.startswith("call_")
        assert len(tc.id) == len("call_") + 12

    def test_auto_id_unique(self) -> None:
        tc1 = ToolCall(name="a")
        tc2 = ToolCall(name="b")
        assert tc1.id != tc2.id

    def test_explicit_id(self) -> None:
        tc = ToolCall(id="my_id", name="calc", arguments={"x": 1})
        assert tc.id == "my_id"

    def test_default_arguments(self) -> None:
        tc = ToolCall(name="noop")
        assert tc.arguments == {}


class TestToolResult:
    def test_creation(self) -> None:
        tr = ToolResult(tool_call_id="c1", content="ok")
        assert tr.tool_call_id == "c1"
        assert tr.content == "ok"
        assert tr.is_error is False

    def test_error_flag(self) -> None:
        tr = ToolResult(tool_call_id="c1", content="boom", is_error=True)
        assert tr.is_error is True


class TestMessageFactories:
    def test_system_message(self) -> None:
        msg = Message.system("You are helpful.")
        assert msg.role == Role.SYSTEM
        assert msg.content == "You are helpful."
        assert msg.created_at is not None

    def test_user_message(self) -> None:
        msg = Message.user("Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_assistant_message_text(self) -> None:
        msg = Message.assistant("Reply")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Reply"
        assert msg.tool_calls is None

    def test_assistant_message_with_tool_calls(self) -> None:
        tc = ToolCall(name="search", arguments={"q": "test"})
        msg = Message.assistant(content=None, tool_calls=[tc])
        assert msg.role == Role.ASSISTANT
        assert msg.content is None
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_tool_message(self) -> None:
        msg = Message.tool(tool_call_id="c1", content="result text")
        assert msg.role == Role.TOOL
        assert msg.content == "result text"
        assert msg.tool_result is not None
        assert msg.tool_result.tool_call_id == "c1"
        assert msg.tool_result.is_error is False

    def test_tool_message_error(self) -> None:
        msg = Message.tool(tool_call_id="c2", content="fail", is_error=True)
        assert msg.tool_result is not None
        assert msg.tool_result.is_error is True

    def test_factory_kwargs_metadata(self) -> None:
        msg = Message.user("hi", metadata={"source": "test"})
        assert msg.metadata["source"] == "test"


class TestHasToolCalls:
    @pytest.mark.parametrize(
        "msg",
        [
            Message.assistant("plain text"),
            Message(role=Role.ASSISTANT, content="x", tool_calls=None),
            Message(role=Role.ASSISTANT, content="x", tool_calls=[]),
        ],
        ids=["assistant_text", "tool_calls_none", "tool_calls_empty"],
    )
    def test_has_tool_calls_false_cases(self, msg: Message) -> None:
        assert msg.has_tool_calls is False

    def test_with_tool_calls(self) -> None:
        tc = ToolCall(name="calc", arguments={})
        msg = Message.assistant(content=None, tool_calls=[tc])
        assert msg.has_tool_calls is True


class TestMessageChunk:
    def test_defaults(self) -> None:
        chunk = MessageChunk()
        assert chunk.delta_content is None
        assert chunk.delta_tool_calls is None
        assert chunk.finish_reason is None

    def test_with_content(self) -> None:
        chunk = MessageChunk(delta_content="hello", finish_reason="stop")
        assert chunk.delta_content == "hello"
        assert chunk.finish_reason == "stop"
