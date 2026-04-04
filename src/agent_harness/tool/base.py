"""Base tool interface and schema for agent_harness.

Tools are the primary way agents interact with the external world.
Every tool exposes a JSON Schema description for LLM function calling.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Description of a single tool parameter."""
    name: str
    type: str  # JSON Schema type: "string", "integer", "number", "boolean", "array", "object"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None


class ToolSchema(BaseModel):
    """JSON Schema-compatible tool description for LLM function calling.

    This is the format passed to LLM providers to describe available tools.
    """
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {
        "type": "object",
        "properties": {},
        "required": [],
    })

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class BaseTool(ABC):
    """Abstract base class for all tools.

    Subclass this to create custom tools, or use the @tool decorator
    for simpler function-based tools.

    Attributes:
        name: Unique tool name (used by LLM to invoke).
        description: Human-readable description (included in LLM prompt).
    """

    name: str
    description: str

    def __init__(
        self,
        name: str,
        description: str,
        *,
        executor_timeout: float | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.executor_timeout = executor_timeout

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool with given arguments.

        Args:
            **kwargs: Tool-specific arguments matching the schema.

        Returns:
            String result to be passed back to the LLM.
        """
        ...

    def get_schema(self) -> ToolSchema:
        """Get the JSON Schema description of this tool.

        Default implementation returns a basic schema.
        Override in subclasses or use @tool decorator for auto-generation.
        """
        return ToolSchema(name=self.name, description=self.description)

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"
