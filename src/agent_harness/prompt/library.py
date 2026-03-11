"""Prompt library for named, versioned prompt templates."""
from __future__ import annotations

from agent_harness.core.registry import Registry
from agent_harness.prompt.template import PromptTemplate


class PromptLibrary:
    """Registry for named and versioned prompt templates.

    Templates are stored with a composite key of name + version,
    allowing multiple versions of the same prompt to coexist.

    Example:
        library = PromptLibrary()
        library.register("researcher", PromptTemplate("You are a researcher..."))
        template = library.get("researcher")
    """

    def __init__(self) -> None:
        self._registry: Registry[PromptTemplate] = Registry()

    def _key(self, name: str, version: str) -> str:
        return f"{name}@{version}"

    def register(
        self, name: str, template: PromptTemplate, version: str = "latest"
    ) -> None:
        """Register a template with a name and optional version."""
        self._registry.register(self._key(name, version), template)

    def get(self, name: str, version: str = "latest") -> PromptTemplate:
        """Retrieve a template by name and version."""
        return self._registry.get(self._key(name, version))

    def has(self, name: str, version: str = "latest") -> bool:
        """Check if a template exists."""
        return self._registry.has(self._key(name, version))

    def list_names(self) -> list[str]:
        """List all registered template keys (name@version)."""
        return self._registry.list_names()

    def __repr__(self) -> str:
        return f"PromptLibrary(templates={self._registry.list_names()})"
