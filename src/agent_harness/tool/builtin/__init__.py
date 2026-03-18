"""Builtin tools shipped with agent_harness."""
from __future__ import annotations

from agent_harness.tool.base import BaseTool
from agent_harness.tool.builtin.file_ops import list_directory, read_file, write_file
from agent_harness.tool.builtin.http_request import http_request
from agent_harness.tool.builtin.python_exec import python_exec
from agent_harness.tool.builtin.web_search import web_search

BUILTIN_TOOLS: list[BaseTool] = [
    read_file,
    write_file,
    list_directory,
    python_exec,
    http_request,
    web_search,
]

__all__ = ["BUILTIN_TOOLS"]
