"""Built-in tools for the agent application."""
from agent_app.tools.filesystem import (
    FILESYSTEM_TOOLS,
    READONLY_FILESYSTEM_TOOLS,
    WRITABLE_FILESYSTEM_TOOLS,
    edit_file,
    glob_files,
    grep_files,
    list_dir,
    read_file,
    write_file,
)
from agent_app.tools.paper_fetch import paper_fetch
from agent_app.tools.paper_search import paper_search
from agent_app.tools.pdf_parser import pdf_parser
from agent_app.tools.skill_tool import skill_tool
from agent_app.tools.take_notes import list_notes, read_notes, take_notes
from agent_app.tools.terminal_tool import terminal_tool
from agent_app.tools.web_fetch import web_fetch
from agent_app.tools.web_search import web_search
from agent_harness.tool.base import BaseTool

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
    skill_tool,
    *FILESYSTEM_TOOLS,
]

__all__ = [
    "BUILTIN_TOOLS",
    "FILESYSTEM_TOOLS",
    "READONLY_FILESYSTEM_TOOLS",
    "WRITABLE_FILESYSTEM_TOOLS",
    "terminal_tool",
    "web_fetch",
    "web_search",
    "pdf_parser",
    "paper_search",
    "paper_fetch",
    "take_notes",
    "list_notes",
    "read_notes",
    "skill_tool",
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "glob_files",
    "grep_files",
]
