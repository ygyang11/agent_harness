"""Resilience patterns for LLM calls: retry, fallback, rate limiting."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, TYPE_CHECKING

from pydantic import BaseModel

from agent_harness.core.errors import LLMError, LLMRateLimitError
from agent_harness.core.message import Message
from agent_harness.llm.types import LLMResponse, StreamDelta
from agent_harness.tool.base import ToolSchema

if TYPE_CHECKING:
    from agent_harness.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class RetryPolicy(BaseModel):
    """Configuration for retry behavior."""
    max_retries: int = 3
    backoff_factor: float = 2.0
    initial_delay: float = 1.0
    max_delay: float = 60.0
    retryable_exceptions: list[str] = [
        "LLMRateLimitError",
        "ConnectionError",
        "TimeoutError",
    ]

    def is_retryable(self, error: Exception) -> bool:
        return type(error).__name__ in self.retryable_exceptions

    def get_delay(self, attempt: int) -> float:
        delay = self.initial_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)


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
                    self._timestamps.append(time.monotonic())
                    return
                # Calculate wait time but release lock before sleeping
                wait_time = self._timestamps[0] + self._window - now

            if wait_time > 0:
                logger.debug("Rate limiter: waiting %.1fs", wait_time)
                await asyncio.sleep(wait_time)


class RetryableLLM:
    """Wraps a BaseLLM with automatic retry on transient failures.

    Example:
        llm = OpenAIProvider(config)
        resilient = RetryableLLM(llm, policy=RetryPolicy(max_retries=3))
        response = await resilient.generate(messages)
    """

    def __init__(
        self,
        llm: BaseLLM,
        policy: RetryPolicy | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.llm = llm
        self.policy = policy or RetryPolicy()
        self.rate_limiter = rate_limiter

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        last_error: Exception | None = None

        for attempt in range(self.policy.max_retries + 1):
            try:
                if self.rate_limiter:
                    await self.rate_limiter.acquire()
                return await self.llm.generate(messages, tools=tools, **kwargs)
            except Exception as e:
                last_error = e
                if not self.policy.is_retryable(e) or attempt >= self.policy.max_retries:
                    raise
                delay = self.policy.get_delay(attempt)
                # Use retry_after from rate limit errors if available
                if isinstance(e, LLMRateLimitError) and e.retry_after:
                    delay = max(delay, e.retry_after)
                logger.warning(
                    "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, self.policy.max_retries, delay, e,
                )
                await asyncio.sleep(delay)

        raise last_error  # type: ignore[misc]


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
                return await provider.generate(messages, tools=tools, **kwargs)
            except Exception as e:
                logger.warning("Provider %s failed: %s, trying next", provider, e)
                errors.append(e)

        raise LLMError(
            f"All {len(self.providers)} providers failed",
            details={"errors": [str(e) for e in errors]},
        )
