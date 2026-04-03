"""Filesystem tools — split into readonly and writable groups for flexible composition."""

from agent_app.tools.filesystem.edit_file import edit_file
from agent_app.tools.filesystem.glob_files import glob_files
from agent_app.tools.filesystem.grep_files import grep_files
from agent_app.tools.filesystem.list_dir import list_dir
from agent_app.tools.filesystem.read_file import read_file
from agent_app.tools.filesystem.write_file import write_file
from agent_harness.tool.base import BaseTool

READONLY_FILESYSTEM_TOOLS: list[BaseTool] = [
    read_file,
    list_dir,
    glob_files,
    grep_files,
]
WRITABLE_FILESYSTEM_TOOLS: list[BaseTool] = [
    write_file,
    edit_file,
]
FILESYSTEM_TOOLS: list[BaseTool] = [
    *READONLY_FILESYSTEM_TOOLS,
    *WRITABLE_FILESYSTEM_TOOLS,
]

__all__ = [
    "FILESYSTEM_TOOLS",
    "READONLY_FILESYSTEM_TOOLS",
    "WRITABLE_FILESYSTEM_TOOLS",
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "glob_files",
    "grep_files",
]
