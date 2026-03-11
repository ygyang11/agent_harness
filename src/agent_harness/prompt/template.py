"""Prompt template with Jinja2 rendering and variable validation."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from jinja2 import BaseLoader, Environment, TemplateSyntaxError


# Shared Jinja2 environment
_jinja_env = Environment(loader=BaseLoader(), keep_trailing_newline=True)


class PromptTemplate:
    """A prompt template with Jinja2 rendering and input variable validation.

    Supports Jinja2 syntax: {{ variable }}, {% if %}, {% for %}, etc.

    Example:
        template = PromptTemplate(
            template="You are a {{ role }}. Answer about {{ topic }}.",
            input_variables=["role", "topic"],
        )
        rendered = template.render(role="researcher", topic="AI safety")
    """

    def __init__(self, template: str, input_variables: list[str] | None = None) -> None:
        self.template = template
        self.input_variables = input_variables or self._extract_variables(template)
        # Validate template syntax
        try:
            self._compiled = _jinja_env.from_string(template)
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid template syntax: {e}") from e

    @staticmethod
    def _extract_variables(template: str) -> list[str]:
        """Extract variable names from a Jinja2 template."""
        # Match {{ var_name }} patterns (simple variable references)
        pattern = r"\{\{\s*(\w+)\s*\}\}"
        return list(dict.fromkeys(re.findall(pattern, template)))

    def render(self, **kwargs: Any) -> str:
        """Render the template with provided variables.

        Args:
            **kwargs: Variable values to fill in.

        Returns:
            Rendered string.

        Raises:
            ValueError: If required variables are missing.
        """
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return self._compiled.render(**kwargs)

    def partial(self, **kwargs: Any) -> PromptTemplate:
        """Create a new template with some variables pre-filled.

        Args:
            **kwargs: Variable values to pre-fill.

        Returns:
            New PromptTemplate with remaining unfilled variables.
        """
        # Render known vars, keep unknown as template vars
        remaining_vars = [v for v in self.input_variables if v not in kwargs]
        rendered = self._compiled.render(**kwargs)
        return PromptTemplate(template=rendered, input_variables=remaining_vars)

    @classmethod
    def from_file(cls, path: str | Path) -> PromptTemplate:
        """Load a template from a file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")
        return cls(template=path.read_text())

    def __repr__(self) -> str:
        preview = self.template[:60] + "..." if len(self.template) > 60 else self.template
        return f"PromptTemplate(vars={self.input_variables}, template={preview!r})"
