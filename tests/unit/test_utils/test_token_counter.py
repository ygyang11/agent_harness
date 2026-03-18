"""Tests for token truncation helpers."""
from __future__ import annotations

from agent_harness.utils import token_counter
from agent_harness.utils.token_counter import count_tokens, truncate_text_by_tokens


class TestTruncateTextByTokens:
    def test_no_truncation_when_within_budget(self) -> None:
        text = "hello world"
        assert truncate_text_by_tokens(text, max_tokens=50) == text

    def test_truncation_with_suffix(self) -> None:
        text = "hello " * 1000
        truncated = truncate_text_by_tokens(text, max_tokens=40, suffix="...")
        assert truncated != text
        assert truncated.endswith("...")
        assert count_tokens(truncated) <= 40

    def test_truncation_without_suffix(self) -> None:
        text = "hello " * 1000
        truncated = truncate_text_by_tokens(text, max_tokens=40, suffix="")
        assert truncated != text
        assert not truncated.endswith("...")
        assert count_tokens(truncated) <= 40

    def test_fallback_without_tiktoken(self, monkeypatch) -> None:
        monkeypatch.setattr(token_counter, "_get_encoding", lambda _name: None)
        text = "x" * 500
        truncated = truncate_text_by_tokens(text, max_tokens=20, suffix="...")
        assert truncated.endswith("...")
        assert len(truncated) <= 60
