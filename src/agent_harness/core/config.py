"""Configuration management for agent_harness.

Supports multiple configuration sources: defaults, YAML files, environment variables.
Environment variables take highest precedence.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for an LLM provider."""

    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 120.0

    def model_post_init(self, __context: Any) -> None:
        # Auto-resolve API keys from environment if not set
        if self.api_key is None:
            env_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
            }
            env_var = env_map.get(self.provider)
            if env_var:
                self.api_key = os.environ.get(env_var)


class ToolConfig(BaseModel):
    """Configuration for the tool execution system."""

    max_concurrency: int = 5
    default_timeout: float = 30.0
    sandbox_enabled: bool = False


class MemoryConfig(BaseModel):
    """Configuration for the memory system."""

    short_term_max_messages: int = 50
    short_term_max_tokens: int = 8000
    short_term_strategy: str = "sliding_window"  # sliding_window | token_limit


class TracingConfig(BaseModel):
    """Configuration for observability."""

    enabled: bool = True
    exporter: str = "console"  # console | json_file
    export_path: str = "./traces"


class HarnessConfig(BaseModel):
    """Root configuration for the agent_harness framework."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    tool: ToolConfig = Field(default_factory=ToolConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    verbose: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> HarnessConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @classmethod
    def from_env(cls) -> HarnessConfig:
        """Build configuration from environment variables.

        Recognized env vars:
            HARNESS_LLM_PROVIDER, HARNESS_LLM_MODEL, HARNESS_LLM_TEMPERATURE,
            HARNESS_LLM_MAX_TOKENS, HARNESS_VERBOSE, HARNESS_TRACING_ENABLED
        """
        data: dict[str, Any] = {}

        llm_data: dict[str, Any] = {}
        if v := os.environ.get("HARNESS_LLM_PROVIDER"):
            llm_data["provider"] = v
        if v := os.environ.get("HARNESS_LLM_MODEL"):
            llm_data["model"] = v
        if v := os.environ.get("HARNESS_LLM_TEMPERATURE"):
            llm_data["temperature"] = float(v)
        if v := os.environ.get("HARNESS_LLM_MAX_TOKENS"):
            llm_data["max_tokens"] = int(v)
        if llm_data:
            data["llm"] = llm_data

        if v := os.environ.get("HARNESS_VERBOSE"):
            data["verbose"] = v.lower() in ("1", "true", "yes")

        tracing_data: dict[str, Any] = {}
        if v := os.environ.get("HARNESS_TRACING_ENABLED"):
            tracing_data["enabled"] = v.lower() in ("1", "true", "yes")
        if tracing_data:
            data["tracing"] = tracing_data

        return cls.model_validate(data)

    def merge(self, other: HarnessConfig) -> HarnessConfig:
        """Merge another config into this one. `other` values take precedence."""
        base = self.model_dump()
        override = other.model_dump(exclude_defaults=True)
        return HarnessConfig.model_validate(_deep_merge(base, override))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
