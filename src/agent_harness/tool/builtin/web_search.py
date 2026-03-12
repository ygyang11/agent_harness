"""Web search tool with Tavily and SerpAPI backends."""
from __future__ import annotations

import logging
import os

from agent_harness.core.config import SearchConfig
from agent_harness.tool.decorator import tool

logger = logging.getLogger(__name__)


def _resolve_config() -> SearchConfig:
    """Resolve SearchConfig (reads from env vars via model_post_init)."""
    provider = os.environ.get("HARNESS_SEARCH_PROVIDER", "tavily")
    return SearchConfig(provider=provider)


async def _search_tavily(query: str, max_results: int, api_key: str) -> str:
    """Search using Tavily API."""
    try:
        from tavily import AsyncTavilyClient  # noqa: PLC0415
    except ImportError:
        return "Error: tavily-python is not installed. Run `pip install tavily-python`."

    client = AsyncTavilyClient(api_key=api_key)
    response = await client.search(query=query, max_results=max_results)

    results = response.get("results", [])
    if not results:
        return f"No results found for: {query}"

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        snippet = r.get("content", "")[:2000]
        url = r.get("url", "")
        lines.append(f"{i}. {title}\n   {snippet}\n   URL: {url}")
    return "\n\n".join(lines)


async def _search_serpapi(query: str, max_results: int, api_key: str) -> str:
    """Search using SerpAPI."""
    try:
        from serpapi import GoogleSearch  # noqa: PLC0415
    except ImportError:
        return "Error: google-search-results is not installed. Run `pip install google-search-results`."

    import asyncio  # noqa: PLC0415

    def _do_search() -> dict:
        search = GoogleSearch({"q": query, "num": max_results, "api_key": api_key})
        return search.get_dict()

    result = await asyncio.get_event_loop().run_in_executor(None, _do_search)
    organic = result.get("organic_results", [])
    if not organic:
        return f"No results found for: {query}"

    lines: list[str] = []
    for i, r in enumerate(organic[:max_results], 1):
        title = r.get("title", "No title")
        snippet = r.get("snippet", "")[:2000]
        url = r.get("link", "")
        lines.append(f"{i}. {title}\n   {snippet}\n   URL: {url}")
    return "\n\n".join(lines)


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information using Tavily or SerpAPI.

    Configure the search provider via config.yaml or environment variables
    (TAVILY_API_KEY, SERPAPI_API_KEY, HARNESS_SEARCH_PROVIDER).

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        Formatted search results with title, snippet, and URL for each result.
    """
    config = _resolve_config()
    provider = config.provider

    if provider == "tavily":
        api_key = config.tavily_api_key or ""
        if not api_key:
            return (
                "Web search not configured: TAVILY_API_KEY not set. "
                "Set the environment variable or configure in config.yaml."
            )
        return await _search_tavily(query, max_results, api_key)
    elif provider == "serpapi":
        api_key = config.serpapi_api_key or ""
        if not api_key:
            return (
                "Web search not configured: SERPAPI_API_KEY not set. "
                "Set the environment variable or configure in config.yaml."
            )
        return await _search_serpapi(query, max_results, api_key)
    else:
        return f"Unknown search provider: {provider!r}. Use 'tavily' or 'serpapi'."
