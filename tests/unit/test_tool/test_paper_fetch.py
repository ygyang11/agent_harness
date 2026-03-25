"""Tests for the paper_fetch builtin tool."""
from __future__ import annotations

from agent_harness.tool.builtin.paper_fetch import (
    _format_metadata,
    paper_fetch,
)


class TestFormatMetadata:
    def test_full_arxiv_metadata(self) -> None:
        paper = {
            "title": "Attention Is All You Need",
            "arxiv_id": "1706.03762",
            "authors": ["Vaswani", "Shazeer", "Parmar"],
            "published": "2017-06-12",
            "abstract": "The dominant sequence transduction models...",
            "categories": ["cs.CL", "cs.AI"],
            "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
            "abs_url": "https://arxiv.org/abs/1706.03762",
        }
        result = _format_metadata(paper)
        assert "# Attention Is All You Need" in result
        assert "1706.03762" in result
        assert "Vaswani" in result
        assert "## Abstract" in result
        assert "paper_fetch" in result

    def test_full_s2_metadata(self) -> None:
        paper = {
            "title": "Test Paper",
            "s2_id": "abc123",
            "doi": "10.1234/xxx",
            "authors": ["Alice"],
            "year": 2023,
            "venue": "NeurIPS",
            "citation_count": 50,
            "reference_count": 30,
            "fields_of_study": ["Computer Science"],
            "publication_types": ["Conference"],
            "abstract": "We propose...",
            "tldr": "A short summary.",
            "pdf_url": "https://example.com/paper.pdf",
        }
        result = _format_metadata(paper)
        assert "# Test Paper" in result
        assert "NeurIPS" in result
        assert "Citations" in result
        assert "References" in result
        assert "Fields of Study" in result
        assert "## TL;DR" in result

    def test_minimal_metadata(self) -> None:
        paper = {"title": "Test"}
        result = _format_metadata(paper)
        assert "# Test" in result

    def test_metadata_actionable_guidance(self) -> None:
        paper = {"title": "X", "arxiv_id": "2301.07041"}
        result = _format_metadata(paper)
        assert 'mode="full"' in result
        assert "paper_fetch" in result

    def test_metadata_no_guidance_without_id(self) -> None:
        paper = {"title": "X"}
        result = _format_metadata(paper)
        assert "paper_fetch" not in result

    def test_publication_date_preferred_over_year(self) -> None:
        paper = {"title": "X", "publication_date": "2023-06-15", "year": 2023}
        result = _format_metadata(paper)
        assert "2023-06-15" in result
        assert "Year" not in result


class TestPaperFetchTool:
    async def test_empty_id(self) -> None:
        result = await paper_fetch.execute(paper_id="")
        assert "Error" in result

    async def test_unknown_mode(self) -> None:
        result = await paper_fetch.execute(paper_id="test", mode="unknown")
        assert "Error" in result
        assert "metadata" in result

    async def test_unknown_source(self) -> None:
        result = await paper_fetch.execute(
            paper_id="test", mode="metadata", source="unknown"
        )
        assert "Error" in result

    def test_schema_params(self) -> None:
        schema = paper_fetch.get_schema()
        assert schema.name == "paper_fetch"
        props = schema.parameters["properties"]
        assert "paper_id" in props
        assert "mode" in props
        assert "source" in props
