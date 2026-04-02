"""LLM module: model provider abstractions."""
from agent_harness.core.config import HarnessConfig, LLMConfig, resolve_llm_config
from agent_harness.llm.anthropic_provider import AnthropicProvider
from agent_harness.llm.base import BaseLLM, FallbackChain, RateLimiter
from agent_harness.llm.openai_provider import OpenAIProvider
from agent_harness.llm.types import FinishReason, LLMResponse, StreamDelta, Usage

_PROVIDER_MAP: dict[str, type[BaseLLM]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def create_llm(
    config: HarnessConfig | LLMConfig | None = None,
    *,
    model_override: str | None = None,
) -> BaseLLM:
    llm_cfg = resolve_llm_config(config)
    if model_override:
        llm_cfg = LLMConfig(
            provider=llm_cfg.provider,
            model=model_override,
            api_key=llm_cfg.api_key,
            base_url=llm_cfg.base_url,
            timeout=llm_cfg.timeout,
            max_retries=llm_cfg.max_retries,
            retry_delay=llm_cfg.retry_delay,
        )
    provider = _PROVIDER_MAP.get(llm_cfg.provider.lower())
    if provider is None:
        supported = ", ".join(sorted(_PROVIDER_MAP))
        raise ValueError(
            f"Unsupported llm.provider: {llm_cfg.provider!r}. Supported: {supported}"
        )
    return provider(llm_cfg)


def LLM(config: HarnessConfig | LLMConfig | None = None) -> BaseLLM:
    return create_llm(config)


__all__ = [
    "BaseLLM",
    "LLMResponse", "Usage", "FinishReason", "StreamDelta",
    "OpenAIProvider", "AnthropicProvider",
    "FallbackChain", "RateLimiter",
    "create_llm", "LLM",
]
