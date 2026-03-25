"""Builtin tools shipped with agent_harness."""
from __future__ import annotations

from agent_harness.tool.base import BaseTool
from agent_harness.tool.builtin.paper_fetch import paper_fetch
from agent_harness.tool.builtin.paper_search import paper_search
from agent_harness.tool.builtin.pdf_parser import pdf_parser
from agent_harness.tool.builtin.take_notes import list_notes, read_notes, take_notes
from agent_harness.tool.builtin.terminal_tool import terminal_tool
from agent_harness.tool.builtin.web_fetch import web_fetch
from agent_harness.tool.builtin.web_search import web_search

BUILTIN_TOOLS: list[BaseTool] = [
    terminal_tool,
    web_fetch,
    web_search,
    pdf_parser,
    paper_search,
    paper_fetch,
    take_notes,
    list_notes,
    read_notes,
]

__all__ = ["BUILTIN_TOOLS"]
