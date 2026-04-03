"""Create new files only."""

from __future__ import annotations

import logging

from agent_app.tools.filesystem._security import (
    is_sensitive_path,
    normalize_path,
    relative_to_workspace,
)
from agent_harness.tool.decorator import tool

logger = logging.getLogger(__name__)

WRITE_FILE_DESCRIPTION = (
    "Writes content to a new file in the filesystem.\n\n"
    "Usage:\n"
    "- This tool creates NEW files only. If the file already exists, "
    "it will return an error — use edit_file to modify existing files\n"
    "- Parent directories are created automatically if they don't exist\n"
    "- ALWAYS prefer editing existing files (with edit_file) over creating new ones "
    "when possible, as this prevents file bloat and builds on existing work"
)


@tool(description=WRITE_FILE_DESCRIPTION)
async def write_file(file_path: str, content: str) -> str:
    """Create a new file.

    Args:
        file_path: Path to the file (absolute or relative to workspace root).
        content: The full content to write to the file.
    """
    try:
        resolved = normalize_path(file_path)
    except ValueError as exc:
        return f"Error: {exc}"

    if resolved.is_dir():
        return f"Error: {file_path} is a directory."

    if is_sensitive_path(resolved):
        return (
            f"Error: Refusing to write to sensitive file: {file_path}. "
            "This file may contain secrets or security-critical configuration."
        )

    if resolved.exists():
        return f"Error: File already exists: {file_path}. Use edit_file to modify existing files."

    rel = relative_to_workspace(resolved)
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except PermissionError:
        return f"Error: Permission denied: {file_path}"
    except OSError as exc:
        return f"Error: {exc}"

    line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"Created {rel} ({line_count} lines)"
