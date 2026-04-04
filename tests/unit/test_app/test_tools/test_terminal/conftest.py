"""Shared fixtures for terminal tool tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _use_tmp_workspace(tmp_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run all terminal tests in a temp workspace."""
    monkeypatch.chdir(tmp_path)
