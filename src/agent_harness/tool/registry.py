"""Tool registry for managing available tools."""
from __future__ import annotations

from agent_harness.core.registry import Registry
from agent_harness.tool.base import BaseTool, ToolSchema


class ToolRegistry:
    """Registry for tool instances.

    Provides tool discovery and schema generation for LLM function calling.

    Example:
        registry = ToolRegistry()
        registry.register(my_tool)
        schemas = registry.get_schemas()  # Pass to LLM
    """

    def __init__(self) -> None:
        self._registry: Registry[BaseTool] = Registry()

    def register(self, tool: BaseTool) -> None:
        """Register a tool by its name."""
        self._registry.register(tool.name, tool)

    def get(self, name: str) -> BaseTool:
        """Get a tool by name. Raises KeyError if not found."""
        return self._registry.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return self._registry.has(name)

    def unregister(self, name: str) -> None:
        """Remove a tool."""
        self._registry.unregister(name)

    def list_tools(self) -> list[BaseTool]:
        """List all registered tools."""
        return list(self._registry.list_all().values())

    def get_schemas(self) -> list[ToolSchema]:
        """Get schemas for all registered tools (for LLM function calling)."""
        return [tool.get_schema() for tool in self.list_tools()]

    def get_openai_schemas(self) -> list[dict]:
        """Get all tool schemas in OpenAI format."""
        return [schema.to_openai_format() for schema in self.get_schemas()]

    def get_anthropic_schemas(self) -> list[dict]:
        """Get all tool schemas in Anthropic format."""
        return [schema.to_anthropic_format() for schema in self.get_schemas()]

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        return self._registry.has(name)

    def __repr__(self) -> str:
        tools = self._registry.list_names()
        return f"ToolRegistry(tools={tools})"
