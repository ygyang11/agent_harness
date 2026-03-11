"""File-system operation tools."""
from __future__ import annotations

import glob as _glob
import os
from pathlib import Path

from agent_harness.tool.decorator import tool


@tool
def read_file(path: str, encoding: str = "utf-8") -> str:
    """Read and return the contents of a file.

    Args:
        path: Absolute or relative path to the file.
        encoding: Text encoding to use when reading the file.

    Returns:
        The file contents as a string, or an error message on failure.
    """
    try:
        return Path(path).read_text(encoding=encoding)
    except FileNotFoundError:
        return f"Error: file not found – {path}"
    except PermissionError:
        return f"Error: permission denied – {path}"
    except Exception as exc:  # noqa: BLE001
        return f"Error reading {path}: {exc}"


@tool
def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """Write content to a file, creating parent directories if needed.

    Args:
        path: Absolute or relative path to the target file.
        content: Text content to write.
        encoding: Text encoding to use when writing the file.

    Returns:
        A success message with bytes written, or an error message on failure.
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)
        return f"Successfully wrote {len(content)} characters to {path}"
    except PermissionError:
        return f"Error: permission denied – {path}"
    except Exception as exc:  # noqa: BLE001
        return f"Error writing {path}: {exc}"


@tool
def list_directory(path: str = ".", pattern: str = "*") -> str:
    """List directory contents matching a glob pattern.

    Args:
        path: Directory path to list.
        pattern: Glob pattern to filter entries.

    Returns:
        A newline-separated list of matching entries (capped at 100),
        or an error message on failure.
    """
    try:
        target = Path(path)
        if not target.is_dir():
            return f"Error: not a directory – {path}"

        entries = sorted(_glob.glob(str(target / pattern)))
        # Make paths relative to *path* for readability
        base = str(target) + os.sep
        names = [e.removeprefix(base) if e.startswith(base) else e for e in entries]

        total = len(names)
        if total > 100:
            names = names[:100]
            names.append(f"... ({total - 100} more entries omitted)")

        return "\n".join(names) if names else "(empty directory)"
    except PermissionError:
        return f"Error: permission denied – {path}"
    except Exception as exc:  # noqa: BLE001
        return f"Error listing {path}: {exc}"
