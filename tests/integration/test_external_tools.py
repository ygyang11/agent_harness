"""Integration tests for external service tools with mocked APIs."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_web_search_missing_api_key() -> None:
    """web_search returns error when API key is missing."""
    from agent_harness.tool.builtin.web_search import web_search

    with patch("agent_harness.core.config.resolve_search_config") as mock_cfg:
        cfg = MagicMock()
        cfg.provider = "tavily"
        cfg.tavily_api_key = None
        cfg.serpapi_api_key = None
        mock_cfg.return_value = cfg

        result = await web_search.execute(query="test query")
        assert "Error" in result or "api" in result.lower()


@pytest.mark.asyncio
async def test_pdf_parser_empty_url() -> None:
    """pdf_parser returns error on empty URL."""
    from agent_harness.tool.builtin.pdf_parser import pdf_parser

    result = await pdf_parser.execute(url="")
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_pdf_parser_whitespace_url() -> None:
    """pdf_parser returns error on whitespace-only URL."""
    from agent_harness.tool.builtin.pdf_parser import pdf_parser

    result = await pdf_parser.execute(url="   ")
    assert result.startswith("Error:")
