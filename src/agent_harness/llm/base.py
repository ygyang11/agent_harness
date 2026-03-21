"""Abstract base class for LLM providers in agent_harness."""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Coroutine, TypeVar

from agent_harness.core.config import LLMConfig
from agent_harness.core.errors import LLMError, LLMRateLimitError
from agent_harness.core.event import EventEmitter
from agent_harness.core.message import Message
from agent_harness.llm.types import LLMResponse, StreamDelta, Usage
from agent_harness.tool.base import ToolSchema

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimiter:
    """Token bucket rate limiter for API calls.

    Limits the number of requests per time window to avoid hitting API limits.
    """

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        self._max_requests = max_requests
        self._window = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        while True:
            async with self._lock:
                now = time.monotonic()
                # Remove timestamps outside the window
                self._timestamps = [
                    t for t in self._timestamps if now - t < self._window
                ]
                if len(self._timestamps) < self._max_requests:
                    self._timestamps.append(now)
                    return
                # Calculate wait time but release lock before sleeping
                wait_time = self._timestamps[0] + self._window - now

            if wait_time > 0:
                logger.debug("Rate limiter: waiting %.1fs", wait_time)
                await asyncio.sleep(wait_time)


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

    def __init__(self, config: LLMConfig, rate_limit: dict[str, Any] | None = None) -> None:
        self.config = config
        self._rate_limiter: RateLimiter | None = None
        if rate_limit:
            self._rate_limiter = RateLimiter(
                max_requests=rate_limit.get("max_requests", 60),
                window_seconds=rate_limit.get("window_seconds", 60.0),
            )

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
        if False:
            yield

    async def generate_with_events(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate with automatic event emission and built-in retry."""
        await self.emit("llm.generate.start", model=self.model_name, message_count=len(messages))
        try:
            if self._rate_limiter:
                await self._rate_limiter.acquire()
            response = await self._with_retry(
                lambda: self.generate(messages, tools=tools, **kwargs)
            )
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

    async def stream_with_events(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamDelta]:
        """Stream with automatic event emission, rate limiting, and retry."""
        await self.emit("llm.stream.start", model=self.model_name, message_count=len(messages))
        try:
            if self._rate_limiter:
                await self._rate_limiter.acquire()

            async def _do_stream() -> AsyncIterator[StreamDelta]:
                return self.stream(messages, tools=tools, **kwargs)

            stream_iter = await self._with_retry(_do_stream)
            total_usage = Usage()

            async for delta in stream_iter:
                if delta.usage:
                    total_usage = total_usage + delta.usage
                yield delta

            await self.emit(
                "llm.stream.end",
                model=self.model_name,
                usage=total_usage.model_dump(),
            )
        except Exception as e:
            await self.emit("llm.stream.error", model=self.model_name, error=str(e))
            raise

    async def _with_retry(self, call: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """Retry a coroutine on transient errors with exponential backoff."""
        max_retries = self.config.max_retries
        delay = self.config.retry_delay

        for attempt in range(max_retries + 1):
            try:
                return await call()
            except (LLMRateLimitError, ConnectionError, TimeoutError) as e:
                if attempt >= max_retries:
                    raise
                wait = delay * (2 ** attempt)
                if isinstance(e, LLMRateLimitError) and e.retry_after:
                    wait = max(wait, e.retry_after)
                logger.warning(
                    "Transient error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, wait, e,
                )
                await asyncio.sleep(wait)

        raise RuntimeError("Unreachable")  # pragma: no cover

    def _resolve_temperature(self, temperature: float | None) -> float:
        return temperature if temperature is not None else self.config.temperature

    def _resolve_max_tokens(self, max_tokens: int | None) -> int:
        return max_tokens if max_tokens is not None else self.config.max_tokens

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model_name}>"


class FallbackChain:
    """Try multiple LLM providers in order, falling back on failure.

    Example:
        chain = FallbackChain([primary_llm, backup_llm])
        response = await chain.generate(messages)
    """

    def __init__(self, providers: list[BaseLLM]) -> None:
        if not providers:
            raise ValueError("FallbackChain requires at least one provider")
        self.providers = providers

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        errors: list[Exception] = []

        for provider in self.providers:
            try:
                return await provider.generate_with_events(messages, tools=tools, **kwargs)
            except Exception as e:
                logger.warning("Provider %s failed: %s, trying next", provider, e)
                errors.append(e)

        raise LLMError(
            f"All {len(self.providers)} providers failed",
            details={"errors": [str(e) for e in errors]},
        )
