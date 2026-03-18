"""HTTP request tool."""
from __future__ import annotations

import json

from agent_harness.tool.decorator import tool
from agent_harness.utils.token_counter import truncate_text_by_tokens

_MAX_BODY_TOKENS = 15000


@tool
async def http_request(
    url: str,
    method: str = "GET",
    headers: str = "",
    body: str = "",
) -> str:
    """Make an HTTP request and return the response.

    Args:
        url: The target URL.
        method: HTTP method (GET, POST, PUT, DELETE, etc.).
        headers: Optional HTTP headers as a JSON-encoded string
            (e.g. '{"Authorization": "Bearer ..."}').
        body: Optional request body string.

    Returns:
        A string containing the HTTP status code and (possibly truncated)
        response body, or an error message on failure.
    """
    try:
        import aiohttp  # noqa: PLC0415
    except ImportError:
        return "Error: aiohttp is not installed. Run `pip install aiohttp`."

    parsed_headers: dict[str, str] = {}
    if headers:
        try:
            parsed_headers = json.loads(headers)
        except json.JSONDecodeError as exc:
            return f"Error: invalid JSON in headers – {exc}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method.upper(),
                url,
                headers=parsed_headers or None,
                data=body or None,
            ) as resp:
                resp_body = await resp.text()
                resp_body = truncate_text_by_tokens(
                    resp_body,
                    max_tokens=_MAX_BODY_TOKENS,
                    suffix="\n... (truncated)",
                )
                return f"Status: {resp.status}\n\n{resp_body}"
    except aiohttp.ClientError as exc:
        return f"HTTP error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"
