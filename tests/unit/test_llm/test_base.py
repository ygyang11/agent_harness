"""Tests for LLM base classes and MockLLM behavior."""
from __future__ import annotations

import pytest

from agent_harness.core.message import Message
from agent_harness.llm.types import FinishReason

from tests.conftest import MockLLM


class TestMockLLMGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_default_response(self) -> None:
        """MockLLM with no queued responses returns the default."""
        llm = MockLLM()
        messages = [Message.user("hello")]
        response = await llm.generate(messages)

        assert response.message.content == "Default mock response"
        assert response.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_generate_returns_enqueued_text(self) -> None:
        """add_text_response queues a response that generate() returns."""
        llm = MockLLM()
        llm.add_text_response("custom reply")

        response = await llm.generate([Message.user("hi")])

        assert response.message.content == "custom reply"
        assert response.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_generate_returns_tool_call(self) -> None:
        """add_tool_call_response queues a tool-call response."""
        llm = MockLLM()
        llm.add_tool_call_response("search", {"query": "test"})

        response = await llm.generate([Message.user("search for test")])

        assert response.has_tool_calls
        assert response.finish_reason == FinishReason.TOOL_CALLS
        assert response.message.tool_calls is not None
        assert response.message.tool_calls[0].name == "search"
        assert response.message.tool_calls[0].arguments == {"query": "test"}


class TestMockLLMCallHistory:
    @pytest.mark.asyncio
    async def test_call_history_is_tracked(self) -> None:
        """Each generate() call is recorded in call_history."""
        llm = MockLLM()
        msg1 = [Message.user("first")]
        msg2 = [Message.user("second")]

        await llm.generate(msg1)
        await llm.generate(msg2)

        assert len(llm.call_history) == 2
        assert llm.call_history[0][0].content == "first"
        assert llm.call_history[1][0].content == "second"

    @pytest.mark.asyncio
    async def test_empty_call_history_initially(self) -> None:
        """Fresh MockLLM has no call history."""
        llm = MockLLM()
        assert llm.call_history == []


class TestMockLLMMultipleResponses:
    @pytest.mark.asyncio
    async def test_responses_consumed_in_order(self) -> None:
        """Multiple enqueued responses are returned in FIFO order."""
        llm = MockLLM()
        llm.add_text_response("first")
        llm.add_text_response("second")
        llm.add_text_response("third")

        r1 = await llm.generate([Message.user("a")])
        r2 = await llm.generate([Message.user("b")])
        r3 = await llm.generate([Message.user("c")])

        assert r1.message.content == "first"
        assert r2.message.content == "second"
        assert r3.message.content == "third"

    @pytest.mark.asyncio
    async def test_falls_back_to_default_after_queue_exhausted(self) -> None:
        """After all queued responses are consumed, returns default."""
        llm = MockLLM()
        llm.add_text_response("only one")

        await llm.generate([Message.user("a")])
        fallback = await llm.generate([Message.user("b")])

        assert fallback.message.content == "Default mock response"
