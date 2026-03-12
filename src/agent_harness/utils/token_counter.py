"""Token counting utilities for agent_harness.

Wraps tiktoken for accurate token counting across different LLM models.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_harness.core.message import Message

logger = logging.getLogger(__name__)

# Model to encoding mapping
_MODEL_ENCODING_MAP: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-5": "o200k_base",
    "gpt-5-mini": "o200k_base",
    # Anthropic models: use cl100k_base as approximation
    "claude-3-opus": "cl100k_base",
    "claude-3-sonnet": "cl100k_base",
    "claude-3-haiku": "cl100k_base",
    "claude-3.5-sonnet": "cl100k_base",
    "claude-4-sonnet": "cl100k_base",
    "claude-4": "cl100k_base",
    "claude-4.5": "cl100k_base",
    "claude-5": "cl100k_base",
    "claude-6": "cl100k_base",
}

# Prefix-based fallback: if exact match fails, try prefix
_PREFIX_ENCODING_MAP: list[tuple[str, str]] = [
    ("gpt-5", "o200k_base"),
    ("gpt-4o", "o200k_base"),
    ("gpt-4", "cl100k_base"),
    ("gpt-3.5", "cl100k_base"),
    ("claude-", "cl100k_base"),
]

DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def _get_encoding(encoding_name: str):
    """Get a tiktoken encoding, cached."""
    try:
        import tiktoken
        return tiktoken.get_encoding(encoding_name)
    except ImportError:
        logger.warning("tiktoken not installed, token counting will use approximation")
        return None


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in a text string for a given model.
    
    Falls back to word-based approximation if tiktoken is unavailable.
    """
    encoding_name = _MODEL_ENCODING_MAP.get(model)
    if encoding_name is None:
        # Try prefix-based matching
        for prefix, enc in _PREFIX_ENCODING_MAP:
            if model.startswith(prefix):
                encoding_name = enc
                break
        else:
            encoding_name = DEFAULT_ENCODING
    encoding = _get_encoding(encoding_name)

    if encoding is not None:
        return len(encoding.encode(text))

    # Fallback: rough approximation (1 token ≈ 4 chars for English, 2 chars for CJK)
    return len(text) // 3


def count_messages_tokens(messages: list[Message], model: str = "gpt-4o") -> int:
    """Count total tokens for a list of messages.
    
    Accounts for message structure overhead (role, separators).
    """
    total = 0
    tokens_per_message = 4  # overhead per message in ChatML format

    for msg in messages:
        total += tokens_per_message
        if msg.content:
            total += count_tokens(msg.content, model)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                total += count_tokens(tc.name, model)
                total += count_tokens(str(tc.arguments), model)
        if msg.tool_result:
            total += count_tokens(msg.tool_result.content, model)

    total += 3  # assistant reply priming
    return total
