"""Tests for agent_harness.tool.registry — ToolRegistry CRUD and schema generation."""
from __future__ import annotations

from typing import Any

import pytest

from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.tool.registry import ToolRegistry


class _DummyTool(BaseTool):
    """Minimal concrete tool for registry tests."""

    async def execute(self, **kwargs: Any) -> str:
        return "dummy"

    def get_schema(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description)


class TestToolRegistry:
    def _make_tool(self, name: str = "my_tool", desc: str = "A tool") -> _DummyTool:
        return _DummyTool(name=name, description=desc)

    def test_register_and_get(self) -> None:
        reg = ToolRegistry()
        t = self._make_tool()
        reg.register(t)
        assert reg.get("my_tool") is t

    def test_get_unknown_raises(self) -> None:
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_registry_membership_contract(self) -> None:
        reg = ToolRegistry()
        t = self._make_tool()
        assert reg.has("my_tool") is False
        assert "my_tool" not in reg
        assert len(reg) == 0
        reg.register(t)
        assert reg.has("my_tool") is True
        assert "my_tool" in reg
        assert len(reg) == 1

    def test_list_tools(self) -> None:
        reg = ToolRegistry()
        t1 = self._make_tool("alpha", "First")
        t2 = self._make_tool("beta", "Second")
        reg.register(t1)
        reg.register(t2)
        tools = reg.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"alpha", "beta"}

    def test_get_schemas(self) -> None:
        reg = ToolRegistry()
        reg.register(self._make_tool("x", "Desc X"))
        schemas = reg.get_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "x"
        assert schemas[0].description == "Desc X"

    def test_unregister(self) -> None:
        reg = ToolRegistry()
        reg.register(self._make_tool("removable"))
        assert reg.has("removable")
        reg.unregister("removable")
        assert not reg.has("removable")

    def test_get_openai_schemas(self) -> None:
        reg = ToolRegistry()
        reg.register(self._make_tool("fn1", "Func one"))
        schemas = reg.get_openai_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "fn1"

    def test_get_anthropic_schemas(self) -> None:
        reg = ToolRegistry()
        reg.register(self._make_tool("fn1", "Func one"))
        schemas = reg.get_anthropic_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "fn1"
        assert "input_schema" in schemas[0]

    def test_repr(self) -> None:
        reg = ToolRegistry()
        reg.register(self._make_tool("t1"))
        r = repr(reg)
        assert "ToolRegistry" in r
        assert "t1" in r
