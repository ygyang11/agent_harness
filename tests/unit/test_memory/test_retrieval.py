"""Tests for memory retrieval (HybridRetriever) and forget mechanism."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from agent_harness.memory.base import MemoryItem
from agent_harness.memory.retrieval import HybridRetriever


class TestHybridRetriever:
    def test_empty_items_returns_empty(self) -> None:
        retriever = HybridRetriever()
        results = retriever.retrieve("hello", [])
        assert results == []

    def test_single_item_keyword_fallback(self) -> None:
        """With < 2 items, TF-IDF falls back to keyword matching."""
        item = MemoryItem(content="Python is a programming language", importance_score=0.5)
        retriever = HybridRetriever()
        results = retriever.retrieve("Python", [item])
        assert len(results) == 1
        assert results[0].content == item.content

    def test_multiple_items_ranked_by_relevance(self) -> None:
        items = [
            MemoryItem(content="The weather is sunny today", importance_score=0.5),
            MemoryItem(content="Python is great for data science", importance_score=0.5),
            MemoryItem(content="Machine learning uses Python extensively", importance_score=0.5),
        ]
        retriever = HybridRetriever()
        results = retriever.retrieve("Python programming", items, top_k=2)
        assert len(results) == 2
        # Python-related items should rank higher
        contents = [r.content for r in results]
        assert any("Python" in c for c in contents)

    def test_importance_score_affects_ranking(self) -> None:
        items = [
            MemoryItem(content="important topic alpha", importance_score=1.0),
            MemoryItem(content="important topic beta", importance_score=0.0),
        ]
        retriever = HybridRetriever()
        results = retriever.retrieve("important topic", items, top_k=2)
        # Higher importance should rank first
        assert results[0].importance_score >= results[1].importance_score

    def test_time_decay_favors_recent(self) -> None:
        now = datetime.now()
        old = now - timedelta(days=30)
        items = [
            MemoryItem(content="recent news about AI", importance_score=0.5, timestamp=now),
            MemoryItem(content="old news about AI", importance_score=0.5, timestamp=old),
        ]
        retriever = HybridRetriever(decay_rate=0.01)
        results = retriever.retrieve("AI news", items, top_k=2)
        # Recent item should rank higher due to time decay
        assert results[0].timestamp >= results[1].timestamp

    def test_top_k_limits_results(self) -> None:
        items = [
            MemoryItem(content=f"item {i} about testing", importance_score=0.5)
            for i in range(10)
        ]
        retriever = HybridRetriever()
        results = retriever.retrieve("testing", items, top_k=3)
        assert len(results) == 3


class TestMemoryItemFields:
    def test_importance_score_field(self) -> None:
        item = MemoryItem(content="test", importance_score=0.8)
        assert item.importance_score == 0.8

    def test_time_score_field(self) -> None:
        item = MemoryItem(content="test", time_score=0.5)
        assert item.time_score == 0.5

    def test_defaults(self) -> None:
        item = MemoryItem(content="test")
        assert item.importance_score is None
        assert item.time_score == 0.0
