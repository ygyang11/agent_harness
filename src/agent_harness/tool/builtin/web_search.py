"""Web search placeholder tool."""
from __future__ import annotations

from agent_harness.tool.decorator import tool


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for a given query.

    This is a placeholder implementation. Inject a concrete search provider
    (e.g. SerpAPI, Tavily, Bing) to enable real web search.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        A message indicating that no search backend is configured.
    """
    return (
        f"Web search backend not configured. "
        f"Inject a search provider. (query={query!r}, max_results={max_results})"
    )
