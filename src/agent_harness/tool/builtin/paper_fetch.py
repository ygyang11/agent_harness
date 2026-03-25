"""Academic paper fetching tool for metadata and full content retrieval."""
from __future__ import annotations

import json as _json
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from agent_harness.core.config import resolve_paper_config
from agent_harness.tool.decorator import tool
from agent_harness.utils.token_counter import truncate_text_by_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperFetchConfig:
    max_full_tokens: int = 15_000
    html_fetch_timeout: int = 30


_CFG = PaperFetchConfig()

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
)

_S2_DETAIL_FIELDS = (
    "paperId,title,authors,abstract,year,venue,"
    "externalIds,openAccessPdf,citationCount,referenceCount,"
    "fieldsOfStudy,publicationDate,publicationTypes,tldr,journal"
)


# ---------------------------------------------------------------------------
# Metadata formatting (shared by arXiv and S2)
# ---------------------------------------------------------------------------


def _format_metadata(paper: dict[str, Any]) -> str:
    lines = [f"# {paper.get('title', 'Untitled')}"]

    if paper.get("arxiv_id"):
        lines.append(f"**arXiv ID**: {paper['arxiv_id']}")
    if paper.get("s2_id"):
        lines.append(f"**S2 ID**: {paper['s2_id']}")
    if paper.get("doi"):
        lines.append(f"**DOI**: {paper['doi']}")

    authors = paper.get("authors", [])
    if authors:
        lines.append(f"**Authors**: {', '.join(authors)}")

    if paper.get("publication_date"):
        lines.append(f"**Publication Date**: {paper['publication_date']}")
    elif paper.get("published"):
        lines.append(f"**Published**: {paper['published']}")
    elif paper.get("year"):
        lines.append(f"**Year**: {paper['year']}")

    if paper.get("venue"):
        lines.append(f"**Venue**: {paper['venue']}")
    if paper.get("categories"):
        lines.append(f"**Categories**: {', '.join(paper['categories'])}")
    if paper.get("fields_of_study"):
        lines.append(f"**Fields of Study**: {', '.join(paper['fields_of_study'])}")
    if paper.get("publication_types"):
        lines.append(f"**Type**: {', '.join(paper['publication_types'])}")
    if paper.get("citation_count") is not None:
        lines.append(f"**Citations**: {paper['citation_count']}")
    if paper.get("reference_count") is not None:
        lines.append(f"**References**: {paper['reference_count']}")

    if paper.get("abstract"):
        lines.append(f"\n## Abstract\n\n{paper['abstract']}")

    if paper.get("tldr"):
        lines.append(f"\n## TL;DR\n\n{paper['tldr']}")

    if paper.get("pdf_url"):
        lines.append(f"\n**PDF**: {paper['pdf_url']}")
    if paper.get("abs_url"):
        lines.append(f"**Page**: {paper['abs_url']}")

    identifier = paper.get("arxiv_id") or paper.get("doi") or paper.get("s2_id") or ""
    if identifier:
        source_hint = "arxiv" if paper.get("arxiv_id") else "semantic_scholar"
        lines.append(
            f"\n---\nTo get the full paper content, use: "
            f'`paper_fetch(paper_id="{identifier}", mode="full", source="{source_hint}")`'
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# arXiv metadata
# ---------------------------------------------------------------------------


async def _fetch_arxiv_metadata(arxiv_id: str) -> str:
    from agent_harness.tool.builtin.paper_search import (
        _fetch_xml,
        _normalize_arxiv_id,
        _parse_arxiv_entry,
    )

    clean_id = _normalize_arxiv_id(arxiv_id)
    ns = "{http://www.w3.org/2005/Atom}"
    url = f"http://export.arxiv.org/api/query?{urlencode({'id_list': clean_id})}"
    root = await _fetch_xml(url)

    entries = root.findall(f"{ns}entry")
    if not entries:
        return f"Error: no arXiv paper found for ID: {clean_id}"

    paper = _parse_arxiv_entry(entries[0])
    return _format_metadata(paper)


# ---------------------------------------------------------------------------
# Semantic Scholar metadata
# ---------------------------------------------------------------------------


async def _fetch_s2_metadata(paper_id: str, api_key: str | None) -> str:
    from agent_harness.tool.builtin.paper_search import (
        _api_get_with_retry,
        _parse_s2_paper,
    )

    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/"
        f"{paper_id}?fields={_S2_DETAIL_FIELDS}"
    )

    extra_headers: dict[str, str] = {}
    if api_key:
        extra_headers["x-api-key"] = api_key

    try:
        status, body = await _api_get_with_retry(url, headers=extra_headers)
    except Exception as exc:
        return f"Error: Semantic Scholar request failed: {exc}"

    if status == 404:
        return (
            f"Error: paper not found: {paper_id}. "
            f"Supported ID formats: DOI:10.xxx, ARXIV:2301.xxxxx, "
            f"ACM:xxx, PMID:xxx, CorpusId:xxx, or S2 paper ID."
        )
    if status == 429:
        return (
            "Error: Semantic Scholar rate limit exceeded after retries. "
            "Try again later, or use source='arxiv' if the paper is on arXiv."
        )
    if status != 200:
        return (
            f"Error: Semantic Scholar API returned HTTP {status}. Detail: {body[:200]}"
        )

    data = _json.loads(body)

    paper = _parse_s2_paper(data)
    paper["publication_date"] = data.get("publicationDate", "")
    paper["reference_count"] = data.get("referenceCount", 0)
    paper["fields_of_study"] = data.get("fieldsOfStudy") or []
    paper["publication_types"] = data.get("publicationTypes") or []
    tldr = data.get("tldr") or {}
    paper["tldr"] = tldr.get("text", "")
    journal = data.get("journal") or {}
    if journal.get("name"):
        paper["venue"] = paper.get("venue") or journal["name"]

    return _format_metadata(paper)


# ---------------------------------------------------------------------------
# Full content retrieval
# ---------------------------------------------------------------------------


async def _fetch_full_content(
    paper_id: str, source: str, api_key: str | None
) -> str:
    if source == "arxiv":
        from agent_harness.tool.builtin.paper_search import _normalize_arxiv_id

        clean_id = _normalize_arxiv_id(paper_id)

        html_result = await _try_arxiv_html(clean_id)
        if html_result is not None:
            return truncate_text_by_tokens(
                html_result,
                max_tokens=_CFG.max_full_tokens,
                suffix="\n... (truncated)",
            )

        pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
        return await _fetch_via_pdf_parser(pdf_url, paper_id=clean_id)

    pdf_url = await _resolve_pdf_url(paper_id, source, api_key)
    if pdf_url.startswith("Error:"):
        return pdf_url
    return await _fetch_via_pdf_parser(pdf_url)


async def _try_arxiv_html(arxiv_id: str) -> str | None:
    import aiohttp

    html_url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                html_url,
                headers={"User-Agent": _USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=_CFG.html_fetch_timeout),
            ) as resp:
                if resp.status != 200:
                    return None
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type.lower():
                    return None
                body = await resp.text()
    except (aiohttp.ClientError, TimeoutError):
        return None

    from agent_harness.tool.builtin.web_fetch import _extract_text_from_html

    text = _extract_text_from_html(body)
    if len(text.strip()) < 500:
        return None
    return text


async def _fetch_via_pdf_parser(
    pdf_url: str, paper_id: str = ""
) -> str:
    from agent_harness.tool.builtin.pdf_parser import pdf_parser as _pdf_parser

    result = await _pdf_parser.execute(url=pdf_url)

    if result.startswith("Error:"):
        msg = f"Error: failed to extract full content from PDF.\nPDF URL: {pdf_url}"
        if paper_id:
            msg += (
                "\nIf you have not retrieved the paper metadata yet, "
                'use `paper_fetch` with mode="metadata" to get the '
                "title, authors, and abstract."
            )
        msg += f"\nUnderlying error: {result}"
        return msg

    return truncate_text_by_tokens(
        result, max_tokens=_CFG.max_full_tokens, suffix="\n... (truncated)"
    )


async def _resolve_pdf_url(
    paper_id: str, source: str, api_key: str | None
) -> str:
    from agent_harness.tool.builtin.paper_search import _api_get_with_retry

    fields = "openAccessPdf,externalIds"
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/"
        f"{paper_id}?fields={fields}"
    )

    extra_headers: dict[str, str] = {}
    if api_key:
        extra_headers["x-api-key"] = api_key

    try:
        status, body = await _api_get_with_retry(url, headers=extra_headers)
    except Exception as exc:
        return f"Error: cannot resolve PDF URL: {exc}"

    if status == 404:
        return f"Error: paper not found: {paper_id}"
    if status != 200:
        return f"Error: cannot resolve PDF URL (HTTP {status})"

    data = _json.loads(body)

    pdf_info: dict[str, str] = data.get("openAccessPdf") or {}
    oa_pdf_url: str = pdf_info.get("url", "")
    if oa_pdf_url:
        return oa_pdf_url

    external: dict[str, str] = data.get("externalIds") or {}
    arxiv_id = external.get("ArXiv")
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    doi = external.get("DOI")
    if doi:
        return await _try_unpaywall(doi)

    return (
        "Error: no open access PDF available for this paper. "
        "The paper may require institutional access or subscription."
    )


async def _try_unpaywall(doi: str) -> str:
    import aiohttp

    url = f"https://api.unpaywall.org/v2/{doi}?email=agent-harness@example.com"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    return f"Error: no open access PDF found for DOI {doi}"
                data = await resp.json()
    except (aiohttp.ClientError, TimeoutError):
        return f"Error: failed to query Unpaywall for DOI {doi}"

    best: dict[str, str] = data.get("best_oa_location") or {}
    oa_url: str = best.get("url_for_pdf", "")
    if oa_url:
        return oa_url

    return f"Error: no open access PDF found for DOI {doi}"


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------


@tool
async def paper_fetch(
    paper_id: str,
    mode: str = "metadata",
    source: str = "arxiv",
) -> str:
    """Fetch academic paper details by ID.

    Two modes: "metadata" returns detailed structured info (title,
    authors, full abstract, citations, fields of study); "full"
    returns the complete paper content from HTML or PDF.

    Args:
        paper_id: arXiv ID like "2301.07041", or DOI:/ARXIV:/ACM:-prefixed ID for S2.
        mode: "metadata" for structured info, or "full" for complete content.
        source: "arxiv" (default) or "semantic_scholar" for IEEE/ACM/ScienceDirect.
    """
    if not paper_id.strip():
        return "Error: paper_id cannot be empty"

    if mode not in ("metadata", "full"):
        return (
            f"Error: unknown mode {mode!r}. "
            f"Use 'metadata' for structured info or 'full' for complete content."
        )

    cfg = resolve_paper_config(None)

    if mode == "metadata":
        if source == "arxiv":
            return await _fetch_arxiv_metadata(paper_id)
        elif source == "semantic_scholar":
            return await _fetch_s2_metadata(paper_id, cfg.semantic_scholar_api_key)
        else:
            return (
                f"Error: unknown source {source!r}. "
                f"Use 'arxiv' or 'semantic_scholar'."
            )

    return await _fetch_full_content(paper_id, source, cfg.semantic_scholar_api_key)
