"""Tests for RateLimiter (moved from retry module to base)."""
from __future__ import annotations

import pytest

from agent_harness.llm.base import RateLimiter


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
