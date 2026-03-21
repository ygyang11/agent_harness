"""LLM request and response types for agent_harness."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.core.message import Message, MessageChunk


class FinishReason(str, Enum):
    """Reason the LLM stopped generating."""
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"


class Usage(BaseModel):
    """Token usage statistics from an LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        """Accumulate usage across multiple calls."""
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class LLMResponse(BaseModel):
    """Response from an LLM provider."""
    message: Message
    usage: Usage = Field(default_factory=Usage)
    finish_reason: FinishReason = FinishReason.STOP
    model: str | None = None  # actual model used (may differ from requested)
    raw_response: Any | None = None  # original provider response for debugging

    @property
    def has_tool_calls(self) -> bool:
        return self.message.has_tool_calls


class StreamDelta(BaseModel):
    """A single chunk in a streaming LLM response."""
    chunk: MessageChunk
    usage: Usage | None = None
    finish_reason: FinishReason | None = None
