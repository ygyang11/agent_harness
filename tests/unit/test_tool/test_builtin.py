"""Tests for builtin tool exports."""
from __future__ import annotations

from agent_harness.tool.builtin import BUILTIN_TOOLS


class TestBuiltinTools:
    def test_builtin_tools_include_web_search(self) -> None:
        names = [t.name for t in BUILTIN_TOOLS]
        assert "web_search" in names

    def test_builtin_tools_include_all_core_tools(self) -> None:
        names = [t.name for t in BUILTIN_TOOLS]
        assert "read_file" in names
        assert "write_file" in names
        assert "list_directory" in names
        assert "python_exec" in names
        assert "http_request" in names
        assert "take_notes" in names
        assert "list_notes" in names
        assert "read_notes" in names

    def test_all_tools_have_schema(self) -> None:
        for t in BUILTIN_TOOLS:
            schema = t.get_schema()
            assert schema.name == t.name
