"""Exact string replacement preserving file format."""

from __future__ import annotations

import difflib
import logging

from agent_app.tools.filesystem._security import (
    detect_text_file,
    is_sensitive_path,
    normalize_path,
    relative_to_workspace,
)
from agent_harness.tool.decorator import tool

logger = logging.getLogger(__name__)
_MAX_DIFF_LINES = 50

EDIT_FILE_DESCRIPTION = (
    "Performs exact string replacements in files.\n\n"
    "Usage:\n"
    "- You must read the file before editing — understand existing content before making changes\n"
    "- When editing text from read_file output, preserve the exact indentation "
    "(tabs/spaces) as it appears in the file. Never include line number prefixes "
    "in old_string or new_string\n"
    "- The old_string must match exactly. If it appears more than once, "
    "include more surrounding context to make it unique, or set replace_all=True\n"
    "- ALWAYS prefer editing existing files over creating new ones\n"
    "- The file's original line-ending style (LF or CRLF) and BOM are preserved automatically"
)


def _generate_diff(old: str, new: str, filename: str) -> str:
    """Generate unified diff between old and new content."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )
    )
    if not diff:
        return "(no changes)"
    if len(diff) > _MAX_DIFF_LINES:
        return "\n".join(diff[:_MAX_DIFF_LINES]) + f"\n... ({len(diff) - _MAX_DIFF_LINES} more)"
    return "\n".join(diff)


@tool(description=EDIT_FILE_DESCRIPTION)
async def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Edit an existing file by replacing exact string matches.

    Args:
        file_path: Path to the file (absolute or relative to workspace root).
        old_string: The exact string to find and replace. Must be non-empty.
        new_string: The replacement string. Must differ from old_string.
        replace_all: If True, replace all occurrences. Default False.
    """
    if not old_string:
        return "Error: old_string cannot be empty. Use write_file to create new files."

    if old_string == new_string:
        return "Error: old_string and new_string are identical. No edit needed."

    try:
        resolved = normalize_path(file_path, must_exist=True)
    except ValueError as exc:
        return f"Error: {exc}"

    if resolved.is_dir():
        return f"Error: {file_path} is a directory."

    if is_sensitive_path(resolved):
        return (
            f"Error: Refusing to edit sensitive file: {file_path}. "
            "This file may contain secrets or security-critical configuration."
        )

    rel = relative_to_workspace(resolved)

    try:
        file_info = detect_text_file(resolved)
    except ValueError as exc:
        return f"Error: {exc}"

    content = file_info.content
    original_newline = file_info.newline

    work_content = content.replace("\r\n", "\n")
    work_old = old_string.replace("\r\n", "\n")
    work_new = new_string.replace("\r\n", "\n")

    count = work_content.count(work_old)

    if count == 0:
        return (
            f"Error: old_string not found in {rel}. "
            "Make sure the string matches exactly (including whitespace and indentation)."
        )

    if count > 1 and not replace_all:
        return (
            f"Error: old_string appears {count} times in {rel}. "
            "Provide more surrounding context to make it unique, "
            "or set replace_all=True to replace all occurrences."
        )

    if replace_all:
        new_content = work_content.replace(work_old, work_new)
    else:
        new_content = work_content.replace(work_old, work_new, 1)

    if original_newline == "\r\n":
        new_content = new_content.replace("\n", "\r\n")

    try:
        resolved.write_text(new_content, encoding=file_info.encoding)
    except OSError as exc:
        return f"Error: {exc}"

    diff = _generate_diff(work_content, new_content.replace("\r\n", "\n"), rel)
    replaced = count if replace_all else 1
    return f"Edited {rel} ({replaced} replacement{'s' if replaced > 1 else ''})\n{diff}"
