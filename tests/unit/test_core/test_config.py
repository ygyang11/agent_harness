"""Tests for agent_harness.core.config — HarnessConfig defaults, from_env, merge."""
from __future__ import annotations

import pytest

from agent_harness.core.config import (
    HarnessConfig,
    LLMConfig,
    MemoryConfig,
    ToolConfig,
    TracingConfig,
)


class TestLLMConfig:
    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096
        assert cfg.timeout == 120.0
        assert cfg.base_url is None

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = LLMConfig()
        assert cfg.api_key == "sk-test"

    def test_api_key_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        cfg = LLMConfig(api_key="sk-explicit")
        assert cfg.api_key == "sk-explicit"

    def test_anthropic_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
        cfg = LLMConfig(provider="anthropic")
        assert cfg.api_key == "ak-test"


class TestToolConfig:
    def test_defaults(self) -> None:
        cfg = ToolConfig()
        assert cfg.max_concurrency == 5
        assert cfg.default_timeout == 30.0
        assert cfg.sandbox_enabled is False


class TestMemoryConfig:
    def test_defaults(self) -> None:
        cfg = MemoryConfig()
        assert cfg.short_term_max_messages == 50
        assert cfg.short_term_max_tokens == 8000
        assert cfg.short_term_strategy == "sliding_window"


class TestTracingConfig:
    def test_defaults(self) -> None:
        cfg = TracingConfig()
        assert cfg.enabled is True
        assert cfg.exporter == "console"
        assert cfg.export_path == "./traces"


class TestHarnessConfig:
    def test_defaults(self) -> None:
        cfg = HarnessConfig()
        assert cfg.verbose is False
        assert cfg.llm.provider == "openai"
        assert cfg.tool.max_concurrency == 5
        assert cfg.memory.short_term_max_messages == 50
        assert cfg.tracing.enabled is True

    def test_from_env_llm_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARNESS_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("HARNESS_LLM_MODEL", "claude-3")
        monkeypatch.setenv("HARNESS_LLM_TEMPERATURE", "0.5")
        monkeypatch.setenv("HARNESS_LLM_MAX_TOKENS", "2048")

        cfg = HarnessConfig.from_env()
        assert cfg.llm.provider == "anthropic"
        assert cfg.llm.model == "claude-3"
        assert cfg.llm.temperature == 0.5
        assert cfg.llm.max_tokens == 2048

    def test_from_env_verbose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARNESS_VERBOSE", "true")
        cfg = HarnessConfig.from_env()
        assert cfg.verbose is True

    def test_from_env_verbose_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARNESS_VERBOSE", "no")
        cfg = HarnessConfig.from_env()
        assert cfg.verbose is False

    def test_from_env_tracing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARNESS_TRACING_ENABLED", "0")
        cfg = HarnessConfig.from_env()
        assert cfg.tracing.enabled is False

    def test_from_env_no_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HARNESS_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("HARNESS_LLM_MODEL", raising=False)
        monkeypatch.delenv("HARNESS_LLM_TEMPERATURE", raising=False)
        monkeypatch.delenv("HARNESS_LLM_MAX_TOKENS", raising=False)
        monkeypatch.delenv("HARNESS_VERBOSE", raising=False)
        monkeypatch.delenv("HARNESS_TRACING_ENABLED", raising=False)
        cfg = HarnessConfig.from_env()
        assert cfg.llm.provider == "openai"

    def test_merge_other_overrides(self) -> None:
        base = HarnessConfig()
        other = HarnessConfig(llm=LLMConfig(model="gpt-3.5"), verbose=True)
        merged = base.merge(other)
        assert merged.llm.model == "gpt-3.5"
        assert merged.verbose is True
        assert merged.llm.provider == "openai"

    def test_merge_preserves_base_when_other_default(self) -> None:
        base = HarnessConfig(llm=LLMConfig(temperature=0.2))
        other = HarnessConfig()
        merged = base.merge(other)
        assert merged.llm.temperature == 0.2

    def test_merge_deep_nested(self) -> None:
        base = HarnessConfig(memory=MemoryConfig(short_term_max_messages=100))
        other = HarnessConfig(memory=MemoryConfig(short_term_max_tokens=4000))
        merged = base.merge(other)
        assert merged.memory.short_term_max_messages == 100
        assert merged.memory.short_term_max_tokens == 4000

    def test_from_yaml_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            HarnessConfig.from_yaml("/nonexistent/config.yaml")
