"""Tests for streaming tool call accumulation in LLM providers."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_harness.core.message import Message, ToolCall
from agent_harness.llm.types import FinishReason, StreamDelta


def _make_openai_chunk(
    *,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


class TestOpenAIStream:

    @pytest.mark.asyncio
    async def test_text_streaming(self) -> None:
        from agent_harness.llm.openai_provider import OpenAIProvider

        chunks = [
            _make_openai_chunk(content="Hello"),
            _make_openai_chunk(content=" world"),
            _make_openai_chunk(finish_reason="stop"),
        ]

        async def fake_create(**kwargs: object) -> _AsyncIter:
            return _AsyncIter(chunks)

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.config = SimpleNamespace(
            model="test", temperature=0.7, max_tokens=100,
            reasoning_effort=None,
        )
        provider._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )
        )
        provider._rate_limiter = None

        deltas: list[StreamDelta] = []
        async for d in provider.stream([Message.user("hi")]):
            deltas.append(d)

        texts = [d.chunk.delta_content for d in deltas if d.chunk.delta_content]
        assert texts == ["Hello", " world"]
        assert deltas[-1].finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_tool_call_accumulation(self) -> None:
        from agent_harness.llm.openai_provider import OpenAIProvider

        tc_chunk_1 = SimpleNamespace(
            index=0,
            id="call_abc",
            function=SimpleNamespace(name="search", arguments='{"q":'),
        )
        tc_chunk_2 = SimpleNamespace(
            index=0,
            id=None,
            function=SimpleNamespace(name=None, arguments='"hello"}'),
        )

        chunks = [
            _make_openai_chunk(tool_calls=[tc_chunk_1]),
            _make_openai_chunk(tool_calls=[tc_chunk_2]),
            _make_openai_chunk(finish_reason="tool_calls"),
        ]

        async def fake_create(**kwargs: object) -> _AsyncIter:
            return _AsyncIter(chunks)

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.config = SimpleNamespace(
            model="test", temperature=0.7, max_tokens=100,
            reasoning_effort=None,
        )
        provider._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )
        )
        provider._rate_limiter = None

        deltas: list[StreamDelta] = []
        async for d in provider.stream([Message.user("hi")]):
            deltas.append(d)

        final = deltas[-1]
        assert final.finish_reason == FinishReason.TOOL_CALLS
        assert final.chunk.delta_tool_calls is not None
        assert len(final.chunk.delta_tool_calls) == 1
        tc = final.chunk.delta_tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "search"
        assert tc.arguments == {"q": "hello"}


class _AsyncIter:
    """Helper to simulate an async iterator from a list."""
    def __init__(self, items: list[object]) -> None:
        self._items = items
        self._index = 0

    def __aiter__(self) -> _AsyncIter:
        return self

    async def __anext__(self) -> object:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item
