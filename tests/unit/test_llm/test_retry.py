"""Tests for agent_harness.llm.retry — RetryPolicy, RateLimiter, RetryableLLM."""
from __future__ import annotations

import pytest

from agent_harness.core.message import Message
from agent_harness.llm.retry import RateLimiter, RetryableLLM, RetryPolicy

from tests.conftest import MockLLM


class TestRetryPolicy:
    def test_default_values(self) -> None:
        """RetryPolicy has sensible defaults."""
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.backoff_factor == 2.0
        assert policy.initial_delay == 1.0
        assert policy.max_delay == 60.0
        assert "LLMRateLimitError" in policy.retryable_exceptions

    def test_is_retryable_matches_exception_name(self) -> None:
        policy = RetryPolicy(retryable_exceptions=["ValueError"])
        assert policy.is_retryable(ValueError("boom")) is True
        assert policy.is_retryable(TypeError("nope")) is False

    def test_get_delay_exponential_backoff(self) -> None:
        policy = RetryPolicy(initial_delay=1.0, backoff_factor=2.0, max_delay=60.0)
        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 4.0
        assert policy.get_delay(3) == 8.0

    def test_get_delay_capped_at_max(self) -> None:
        policy = RetryPolicy(initial_delay=1.0, backoff_factor=10.0, max_delay=50.0)
        assert policy.get_delay(5) == 50.0


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_succeeds_within_limit(self) -> None:
        """RateLimiter allows requests within the configured rate."""
        limiter = RateLimiter(max_requests=10, window_seconds=60.0)
        # Should not block or raise
        await limiter.acquire()
        await limiter.acquire()

    @pytest.mark.asyncio
    async def test_acquire_tracks_timestamps(self) -> None:
        """Each acquire() records a timestamp."""
        limiter = RateLimiter(max_requests=5, window_seconds=60.0)
        for _ in range(3):
            await limiter.acquire()
        assert len(limiter._timestamps) == 3


class TestRetryableLLM:
    @pytest.mark.asyncio
    async def test_generate_delegates_to_wrapped_llm(self) -> None:
        """RetryableLLM forwards generate() to the underlying LLM."""
        llm = MockLLM()
        llm.add_text_response("success")

        retryable = RetryableLLM(llm=llm, policy=RetryPolicy(max_retries=2))
        response = await retryable.generate([Message.user("test")])

        assert response.message.content == "success"
        assert len(llm.call_history) == 1

    @pytest.mark.asyncio
    async def test_generate_with_rate_limiter(self) -> None:
        """RetryableLLM respects rate limiter when configured."""
        llm = MockLLM()
        llm.add_text_response("ok")

        limiter = RateLimiter(max_requests=100, window_seconds=60.0)
        retryable = RetryableLLM(llm=llm, policy=RetryPolicy(), rate_limiter=limiter)
        response = await retryable.generate([Message.user("test")])

        assert response.message.content == "ok"
        assert len(limiter._timestamps) == 1
