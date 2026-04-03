"""Shared fixtures for filesystem tool tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

_PATCH_TARGETS = [
    "agent_app.tools.filesystem._security.get_workspace_root",
    "agent_app.tools.filesystem.list_dir.get_workspace_root",
    "agent_app.tools.filesystem.glob_files.get_workspace_root",
    "agent_app.tools.filesystem.grep_files.get_workspace_root",
]


@pytest.fixture(autouse=True)
def _workspace_root(tmp_path: Path) -> None:  # type: ignore[misc]
    """Patch get_workspace_root to return tmp_path everywhere it's imported."""
    patches = [patch(target, return_value=tmp_path) for target in _PATCH_TARGETS]
    for p in patches:
        p.start()
    yield
    for p in reversed(patches):
        p.stop()
