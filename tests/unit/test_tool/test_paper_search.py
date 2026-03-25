"""Tests for the paper_search builtin tool."""
from __future__ import annotations

from xml.etree.ElementTree import fromstring

from agent_harness.tool.builtin.paper_search import (
    _build_arxiv_query_url,
    _format_paper_results,
    _looks_like_arxiv_id,
    _normalize_arxiv_id,
    _parse_arxiv_entry,
    _parse_s2_paper,
    paper_search,
)


class TestArxivIdParsing:
    def test_new_style_id(self) -> None:
        assert _looks_like_arxiv_id("2301.07041")

    def test_new_style_with_version(self) -> None:
        assert _looks_like_arxiv_id("2301.07041v2")

    def test_old_style_id(self) -> None:
        assert _looks_like_arxiv_id("cs/0601001")

    def test_url_stripped(self) -> None:
        assert _looks_like_arxiv_id("https://arxiv.org/abs/2301.07041")

    def test_pdf_url_stripped(self) -> None:
        assert _looks_like_arxiv_id("https://arxiv.org/pdf/2301.07041.pdf")

    def test_not_arxiv_id(self) -> None:
        assert not _looks_like_arxiv_id("transformer attention")

    def test_normalize_strips_version(self) -> None:
        assert _normalize_arxiv_id("2301.07041v3") == "2301.07041"

    def test_normalize_strips_url(self) -> None:
        assert _normalize_arxiv_id("https://arxiv.org/abs/2301.07041") == "2301.07041"

    def test_normalize_strips_pdf_url(self) -> None:
        result = _normalize_arxiv_id("https://arxiv.org/pdf/2301.07041.pdf")
        assert result == "2301.07041"


class TestArxivQueryUrl:
    def test_keyword_search(self) -> None:
        url = _build_arxiv_query_url("transformer attention", 10)
        assert "search_query=" in url
        assert "sortBy=relevance" in url

    def test_id_search(self) -> None:
        url = _build_arxiv_query_url("2301.07041", 1)
        assert "id_list=2301.07041" in url
        assert "search_query" not in url

    def test_explicit_id_prefix(self) -> None:
        url = _build_arxiv_query_url("id:2301.07041", 1)
        assert "id_list=2301.07041" in url


class TestArxivParsing:
    def test_parse_entry(self) -> None:
        xml = (
            '<entry xmlns="http://www.w3.org/2005/Atom"'
            '       xmlns:arxiv="http://arxiv.org/schemas/atom">'
            "<id>http://arxiv.org/abs/2301.07041v1</id>"
            "<title>Test Paper Title</title>"
            "<summary>This is the abstract.</summary>"
            "<author><name>Alice</name></author>"
            "<author><name>Bob</name></author>"
            "<published>2023-01-17T00:00:00Z</published>"
            "<updated>2023-01-18T00:00:00Z</updated>"
            '<arxiv:primary_category term="cs.AI"/>'
            '<category term="cs.AI"/>'
            '<category term="cs.CL"/>'
            "</entry>"
        )
        entry = fromstring(xml)
        result = _parse_arxiv_entry(entry)
        assert result["arxiv_id"] == "2301.07041"
        assert result["title"] == "Test Paper Title"
        assert result["authors"] == ["Alice", "Bob"]
        assert result["abstract"] == "This is the abstract."
        assert result["published"] == "2023-01-17"
        assert result["pdf_url"] == "https://arxiv.org/pdf/2301.07041.pdf"

    def test_parse_entry_no_authors(self) -> None:
        xml = (
            '<entry xmlns="http://www.w3.org/2005/Atom">'
            "<id>http://arxiv.org/abs/2301.00001v1</id>"
            "<title>No Authors</title>"
            "<summary>Abstract.</summary>"
            "<published>2023-01-01T00:00:00Z</published>"
            "<updated>2023-01-01T00:00:00Z</updated>"
            "</entry>"
        )
        entry = fromstring(xml)
        result = _parse_arxiv_entry(entry)
        assert result["authors"] == []


class TestS2Parsing:
    def test_parse_s2_paper(self) -> None:
        raw = {
            "paperId": "abc123",
            "title": "A Great Paper",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "abstract": "We propose ...",
            "year": 2023,
            "venue": "NeurIPS",
            "externalIds": {"DOI": "10.1234/xxx", "ArXiv": "2301.07041"},
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
            "citationCount": 42,
            "publicationTypes": ["Conference"],
        }
        result = _parse_s2_paper(raw)
        assert result["s2_id"] == "abc123"
        assert result["title"] == "A Great Paper"
        assert result["authors"] == ["Alice", "Bob"]
        assert result["doi"] == "10.1234/xxx"
        assert result["arxiv_id"] == "2301.07041"
        assert result["citation_count"] == 42
        assert result["pdf_url"] == "https://example.com/paper.pdf"

    def test_parse_s2_paper_missing_fields(self) -> None:
        raw = {"paperId": "abc", "title": "Minimal"}
        result = _parse_s2_paper(raw)
        assert result["s2_id"] == "abc"
        assert result["authors"] == []
        assert result["abstract"] == ""
        assert result["doi"] == ""


class TestFormatResults:
    def test_arxiv_format_includes_all_fields(self) -> None:
        papers = [
            {
                "arxiv_id": "2301.07041",
                "title": "Test",
                "authors": ["Alice"],
                "abstract": "Abstract text",
                "published": "2023-01-17",
                "categories": ["cs.AI"],
                "pdf_url": "https://arxiv.org/pdf/2301.07041.pdf",
                "abs_url": "https://arxiv.org/abs/2301.07041",
            }
        ]
        result = _format_paper_results(papers, source="arxiv")
        assert "2301.07041" in result
        assert "Alice" in result
        assert "Abstract text" in result
        assert "paper_fetch" in result

    def test_s2_format_includes_venue_and_citations(self) -> None:
        papers = [
            {
                "s2_id": "abc",
                "title": "Test",
                "authors": ["Bob"],
                "abstract": "",
                "year": 2023,
                "venue": "ICML",
                "doi": "10.1234/xxx",
                "citation_count": 100,
                "pdf_url": "",
            }
        ]
        result = _format_paper_results(papers, source="semantic_scholar")
        assert "ICML" in result
        assert "100" in result
        assert "DOI" in result

    def test_arxiv_footer_suggests_semantic_scholar(self) -> None:
        papers = [{"title": "X", "authors": [], "abstract": ""}]
        result = _format_paper_results(papers, source="arxiv")
        assert "semantic_scholar" in result

    def test_many_authors_truncated(self) -> None:
        papers = [
            {
                "title": "X",
                "authors": ["A", "B", "C", "D", "E", "F", "G"],
                "abstract": "",
            }
        ]
        result = _format_paper_results(papers, source="arxiv")
        assert "et al." in result
        assert "7 authors" in result


class TestPaperSearchTool:
    async def test_empty_query(self) -> None:
        result = await paper_search.execute(query="")
        assert "Error" in result

    async def test_unknown_source(self) -> None:
        result = await paper_search.execute(query="test", source="unknown")
        assert "Error" in result
        assert "semantic_scholar" in result

    def test_schema_has_correct_params(self) -> None:
        schema = paper_search.get_schema()
        assert schema.name == "paper_search"
        props = schema.parameters["properties"]
        assert "query" in props
        assert "source" in props
        assert "max_results" in props
