"""Configuration management for agent_harness.

Supports multiple configuration sources: defaults, YAML files, environment variables.
Environment variables take highest precedence.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import yaml
from pydantic import BaseModel, Field, PrivateAttr, field_validator

if TYPE_CHECKING:
    from agent_harness.agent.hooks import DefaultHooks


class _EnvVars:
    """Central registry of environment variable names."""
    OPENAI_API_KEY = "OPENAI_API_KEY"
    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
    TAVILY_API_KEY = "TAVILY_API_KEY"
    SERPAPI_API_KEY = "SERPAPI_API_KEY"
    MINERU_API_KEY = "MINERU_API_KEY"
    PADDLEOCR_API_KEY = "PADDLEOCR_API_KEY"
    HARNESS_LLM_PROVIDER = "HARNESS_LLM_PROVIDER"
    HARNESS_LLM_MODEL = "HARNESS_LLM_MODEL"
    HARNESS_LLM_TEMPERATURE = "HARNESS_LLM_TEMPERATURE"
    HARNESS_LLM_MAX_TOKENS = "HARNESS_LLM_MAX_TOKENS"
    HARNESS_LLM_BASE_URL = "HARNESS_LLM_BASE_URL"
    HARNESS_VERBOSE = "HARNESS_VERBOSE"
    HARNESS_TRACING_ENABLED = "HARNESS_TRACING_ENABLED"
    SEMANTIC_SCHOLAR_API_KEY = "SEMANTIC_SCHOLAR_API_KEY"


class LLMConfig(BaseModel):
    """Configuration for an LLM provider."""

    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0
    reasoning_effort: str | None = None

    @field_validator("api_key", "base_url", "reasoning_effort", mode="before")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    def model_post_init(self, __context: Any) -> None:
        # Auto-resolve API keys from environment if not set
        if self.api_key is None:
            env_map = {
                "openai": _EnvVars.OPENAI_API_KEY,
                "anthropic": _EnvVars.ANTHROPIC_API_KEY,
            }
            env_var = env_map.get(self.provider)
            if env_var:
                self.api_key = os.environ.get(env_var)
        # Auto-resolve base_url from environment if not set
        if self.base_url is None:
            self.base_url = os.environ.get(_EnvVars.HARNESS_LLM_BASE_URL)


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
    forget_threshold: float = 0.3


class SearchConfig(BaseModel):
    """Configuration for search providers."""

    provider: str = "tavily"  # "tavily" | "serpapi"
    tavily_api_key: str | None = None
    serpapi_api_key: str | None = None

    @field_validator("tavily_api_key", "serpapi_api_key", mode="before")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.tavily_api_key is None:
            self.tavily_api_key = os.environ.get(_EnvVars.TAVILY_API_KEY)
        if self.serpapi_api_key is None:
            self.serpapi_api_key = os.environ.get(_EnvVars.SERPAPI_API_KEY)


class PdfConfig(BaseModel):
    """Configuration for PDF parsing providers."""

    provider: str = "mineru"  # "mineru" | "paddleocr"
    mineru_api_key: str | None = None
    paddleocr_api_key: str | None = None

    @field_validator("mineru_api_key", "paddleocr_api_key", mode="before")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.mineru_api_key is None:
            self.mineru_api_key = os.environ.get(_EnvVars.MINERU_API_KEY)
        if self.paddleocr_api_key is None:
            self.paddleocr_api_key = os.environ.get(_EnvVars.PADDLEOCR_API_KEY)


class PaperConfig(BaseModel):
    """Configuration for academic paper tools."""

    semantic_scholar_api_key: str | None = None

    @field_validator("semantic_scholar_api_key", mode="before")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.semantic_scholar_api_key is None:
            self.semantic_scholar_api_key = os.environ.get(
                _EnvVars.SEMANTIC_SCHOLAR_API_KEY
            )


class TracingConfig(BaseModel):
    """Configuration for observability."""

    enabled: bool = True
    exporter: str = "console"  # console | json_file
    export_path: str = "./traces"


class SkillConfig(BaseModel):
    """Configuration for the skill system."""

    dirs: list[str] = Field(default_factory=lambda: ["skills"])

    @field_validator("dirs")
    @classmethod
    def _validate_dirs(cls, v: list[str]) -> list[str]:
        for d in v:
            if Path(d).name != "skills":
                raise ValueError(f"Skill directory must end with 'skills', got: {d!r}")
        return v


class HarnessConfig(BaseModel):
    """Root configuration for the agent_harness framework."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    tool: ToolConfig = Field(default_factory=ToolConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    pdf: PdfConfig = Field(default_factory=PdfConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    skill: SkillConfig = Field(default_factory=SkillConfig)
    verbose: bool = False

    _instance: ClassVar[HarnessConfig | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _runtime_hooks: DefaultHooks | None = PrivateAttr(default=None)

    @classmethod
    def get(cls) -> HarnessConfig:
        """Return the active config, or a default instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def from_yaml(cls, path: str | Path) -> HarnessConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        config = cls.model_validate(data)
        cls._instance = config
        return config

    @classmethod
    def from_env(cls) -> HarnessConfig:
        """Build configuration from environment variables.

        Recognized env vars:
            HARNESS_LLM_PROVIDER, HARNESS_LLM_MODEL, HARNESS_LLM_TEMPERATURE,
            HARNESS_LLM_MAX_TOKENS, HARNESS_LLM_BASE_URL,
            HARNESS_VERBOSE, HARNESS_TRACING_ENABLED
        """
        data: dict[str, Any] = {}

        llm_data: dict[str, Any] = {}
        if v := os.environ.get(_EnvVars.HARNESS_LLM_PROVIDER):
            llm_data["provider"] = v
        if v := os.environ.get(_EnvVars.HARNESS_LLM_MODEL):
            llm_data["model"] = v
        if v := os.environ.get(_EnvVars.HARNESS_LLM_TEMPERATURE):
            llm_data["temperature"] = float(v)
        if v := os.environ.get(_EnvVars.HARNESS_LLM_MAX_TOKENS):
            llm_data["max_tokens"] = int(v)
        if v := os.environ.get(_EnvVars.HARNESS_LLM_BASE_URL):
            llm_data["base_url"] = v
        if llm_data:
            data["llm"] = llm_data

        if v := os.environ.get(_EnvVars.HARNESS_VERBOSE):
            data["verbose"] = v.lower() in ("1", "true", "yes")

        tracing_data: dict[str, Any] = {}
        if v := os.environ.get(_EnvVars.HARNESS_TRACING_ENABLED):
            tracing_data["enabled"] = v.lower() in ("1", "true", "yes")
        if tracing_data:
            data["tracing"] = tracing_data

        return cls.model_validate(data)

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        env_override: bool = True,
    ) -> HarnessConfig:
        """Load from YAML and optionally override with environment variables."""
        with cls._lock:
            file_cfg = cls.from_yaml(path) if path is not None else cls()
            if not env_override:
                cls._instance = file_cfg
                return file_cfg

            env_cfg = cls.from_env()
            merged = file_cfg.merge(env_cfg)
            cls._instance = merged
            return merged

    def merge(self, other: HarnessConfig) -> HarnessConfig:
        """Merge another config into this one. `other` values take precedence."""
        base = self.model_dump()
        override = other.model_dump(exclude_defaults=True)
        return HarnessConfig.model_validate(_deep_merge(base, override))

    def get_runtime_hooks(self) -> DefaultHooks | None:
        return self._runtime_hooks

    def set_runtime_hooks(self, hooks: DefaultHooks) -> None:
        self._runtime_hooks = hooks


def resolve_llm_config(config: HarnessConfig | LLMConfig | None) -> LLMConfig:
    if isinstance(config, HarnessConfig):
        return config.llm
    if isinstance(config, LLMConfig):
        return config
    return HarnessConfig.get().llm


def resolve_tool_config(config: HarnessConfig | ToolConfig | None) -> ToolConfig:
    if isinstance(config, HarnessConfig):
        return config.tool
    if isinstance(config, ToolConfig):
        return config
    return HarnessConfig.get().tool


def resolve_search_config(config: HarnessConfig | SearchConfig | None) -> SearchConfig:
    if isinstance(config, HarnessConfig):
        return config.search
    if isinstance(config, SearchConfig):
        return config
    return HarnessConfig.get().search


def resolve_pdf_config(config: HarnessConfig | PdfConfig | None) -> PdfConfig:
    if isinstance(config, HarnessConfig):
        return config.pdf
    if isinstance(config, PdfConfig):
        return config
    return HarnessConfig.get().pdf


def resolve_paper_config(config: HarnessConfig | PaperConfig | None) -> PaperConfig:
    if isinstance(config, HarnessConfig):
        return config.paper
    if isinstance(config, PaperConfig):
        return config
    return HarnessConfig.get().paper


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
