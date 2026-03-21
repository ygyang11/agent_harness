"""Tests for web_fetch tool."""
from __future__ import annotations

import pytest

from agent_harness.tool.builtin.web_fetch import (
    _extract_text_from_html,
    _format_response,
    _is_binary_content_type,
    _is_pdf,
    web_fetch,
)


class TestWebFetchValidation:
    @pytest.mark.asyncio
    async def test_empty_url_returns_error(self) -> None:
        result = await web_fetch.execute(url="")
        assert result.startswith("Error:")
        assert "empty" in result

    @pytest.mark.asyncio
    async def test_file_scheme_is_blocked(self) -> None:
        result = await web_fetch.execute(url="file:///etc/passwd")
        assert result.startswith("Error:")
        assert "unsupported URL scheme" in result

    @pytest.mark.asyncio
    async def test_ftp_scheme_is_blocked(self) -> None:
        result = await web_fetch.execute(url="ftp://example.com/file")
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_missing_host_returns_error(self) -> None:
        result = await web_fetch.execute(url="http://")
        assert result.startswith("Error:")
        assert "missing host" in result

    @pytest.mark.asyncio
    async def test_invalid_timeout_returns_error(self) -> None:
        result = await web_fetch.execute(url="https://example.com", timeout=0)
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_no_scheme_returns_error(self) -> None:
        result = await web_fetch.execute(url="not-a-url")
        assert result.startswith("Error:")


class TestTextExtractor:
    def test_extracts_visible_text(self) -> None:
        html = "<html><body><h1>Title</h1><p>Hello world</p></body></html>"
        result = _extract_text_from_html(html)
        assert "Title" in result
        assert "Hello world" in result

    def test_strips_script_and_style(self) -> None:
        html = "<script>alert(1)</script><style>.x{}</style><p>visible</p>"
        result = _extract_text_from_html(html)
        assert "alert" not in result
        assert ".x" not in result
        assert "visible" in result

    def test_strips_noscript(self) -> None:
        html = "<noscript>hidden</noscript><p>shown</p>"
        result = _extract_text_from_html(html)
        assert "hidden" not in result
        assert "shown" in result

    def test_collapses_blank_lines(self) -> None:
        html = "<p>a</p><br><br><br><p>b</p>"
        result = _extract_text_from_html(html)
        assert "\n\n\n" not in result
        assert "a" in result
        assert "b" in result

    def test_empty_html(self) -> None:
        assert _extract_text_from_html("") == ""

    def test_nested_skip_tags(self) -> None:
        html = "<script><script>inner</script></script><p>ok</p>"
        result = _extract_text_from_html(html)
        assert "inner" not in result
        assert "ok" in result


class TestFormatResponse:
    def test_json_is_pretty_printed(self) -> None:
        result = _format_response('{"key":"value"}', "application/json")
        assert '"key": "value"' in result

    def test_invalid_json_returns_raw(self) -> None:
        result = _format_response("{broken", "application/json")
        assert result == "{broken"

    def test_html_is_extracted(self) -> None:
        result = _format_response(
            "<html><body><p>Hello</p></body></html>",
            "text/html; charset=utf-8",
        )
        assert "<html>" not in result
        assert "Hello" in result

    def test_plain_text_is_passthrough(self) -> None:
        assert _format_response("raw content", "text/plain") == "raw content"

    def test_empty_content_type_is_passthrough(self) -> None:
        assert _format_response("some data", "") == "some data"


class TestBinaryContentTypeDetection:
    def test_pdf_is_not_binary(self) -> None:
        assert _is_binary_content_type("application/pdf") is False

    def test_zip_is_binary(self) -> None:
        assert _is_binary_content_type("application/zip") is True

    def test_octet_stream_is_binary(self) -> None:
        assert _is_binary_content_type("application/octet-stream") is True

    def test_image_png_is_binary(self) -> None:
        assert _is_binary_content_type("image/png") is True

    def test_video_mp4_is_binary(self) -> None:
        assert _is_binary_content_type("video/mp4") is True

    def test_html_is_not_binary(self) -> None:
        assert _is_binary_content_type("text/html; charset=utf-8") is False

    def test_json_is_not_binary(self) -> None:
        assert _is_binary_content_type("application/json") is False

    def test_plain_text_is_not_binary(self) -> None:
        assert _is_binary_content_type("text/plain") is False

    def test_empty_is_not_binary(self) -> None:
        assert _is_binary_content_type("") is False


class TestPdfDetection:
    def test_pdf_content_type(self) -> None:
        assert _is_pdf("application/pdf", "https://example.com/report") is True

    def test_pdf_content_type_with_charset(self) -> None:
        assert _is_pdf("application/pdf; charset=utf-8", "https://example.com/r") is True

    def test_pdf_url_suffix(self) -> None:
        assert _is_pdf("application/octet-stream", "https://example.com/report.pdf") is True

    def test_pdf_url_suffix_case_insensitive(self) -> None:
        assert _is_pdf("application/octet-stream", "https://example.com/Report.PDF") is True

    def test_pdf_url_suffix_with_query_params(self) -> None:
        assert _is_pdf("", "https://cdn.example.com/file.pdf?token=abc") is True

    def test_pdf_empty_content_type_with_pdf_suffix(self) -> None:
        assert _is_pdf("", "https://example.com/doc.pdf") is True

    def test_html_not_pdf(self) -> None:
        assert _is_pdf("text/html", "https://example.com/page") is False

    def test_octet_stream_without_pdf_suffix(self) -> None:
        assert _is_pdf("application/octet-stream", "https://example.com/data.bin") is False

    def test_json_not_pdf(self) -> None:
        assert _is_pdf("application/json", "https://api.example.com/data") is False
