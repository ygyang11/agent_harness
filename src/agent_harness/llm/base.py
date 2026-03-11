"""Abstract base class for LLM providers in agent_harness."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from agent_harness.core.config import LLMConfig
from agent_harness.core.event import EventEmitter
from agent_harness.core.message import Message
from agent_harness.llm.types import LLMResponse, StreamDelta, Usage
from agent_harness.tool.base import ToolSchema

logger = logging.getLogger(__name__)


class BaseLLM(ABC, EventEmitter):
    """Abstract base class for LLM providers.

    All LLM providers (OpenAI, Anthropic, etc.) implement this interface,
    giving agents a uniform way to interact with any model.

    Features:
        - Unified generate/stream interface
        - Tool/function calling support
        - Event emission for observability
        - Token counting
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @property
    def model_name(self) -> str:
        return self.config.model

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: Conversation history.
            tools: Available tools for function calling.
            tool_choice: Tool selection strategy:
                - "auto": LLM decides whether to call a tool
                - "required": LLM must call a tool
                - "none": No tool calling
                - specific name: Force calling a specific tool
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.

        Returns:
            LLMResponse with the generated message and metadata.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamDelta]:
        """Stream a response from the LLM.

        Args:
            Same as generate().

        Yields:
            StreamDelta chunks as they arrive.
        """
        ...
        # Make this a generator
        if False:
            yield  # type: ignore[misc]

    async def generate_with_events(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate with automatic event emission."""
        await self.emit("llm.generate.start", model=self.model_name, message_count=len(messages))
        try:
            response = await self.generate(messages, tools=tools, **kwargs)
            await self.emit(
                "llm.generate.end",
                model=self.model_name,
                usage=response.usage.model_dump(),
                finish_reason=response.finish_reason.value,
            )
            return response
        except Exception as e:
            await self.emit("llm.generate.error", model=self.model_name, error=str(e))
            raise

    def _resolve_temperature(self, temperature: float | None) -> float:
        return temperature if temperature is not None else self.config.temperature

    def _resolve_max_tokens(self, max_tokens: int | None) -> int:
        return max_tokens if max_tokens is not None else self.config.max_tokens

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model_name}>"
