"""Regression tests for RateLimiter lock fix."""
from __future__ import annotations

import asyncio
import time
import pytest

from agent_harness.llm.base import RateLimiter


class TestRateLimiterLockFix:
    """Issue #7: RateLimiter must not hold the lock while sleeping."""

    @pytest.mark.asyncio
    async def test_concurrent_acquires_not_serialized(self) -> None:
        """Multiple tasks should be able to check the rate limiter concurrently."""
        limiter = RateLimiter(max_requests=10, window_seconds=60.0)

        acquired: list[float] = []

        async def worker() -> None:
            await limiter.acquire()
            acquired.append(time.monotonic())

        # 5 concurrent acquires should all complete nearly instantly
        start = time.monotonic()
        await asyncio.gather(*(worker() for _ in range(5)))
        elapsed = time.monotonic() - start

        assert len(acquired) == 5
        assert elapsed < 1.0  # should be very fast

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self) -> None:
        """Requests beyond max_requests should wait."""
        limiter = RateLimiter(max_requests=2, window_seconds=0.5)

        await limiter.acquire()
        await limiter.acquire()

        # Third acquire should need to wait ~0.5s
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        assert elapsed >= 0.3  # waited for window
