"""Anthropic LLM provider for agent_harness (Claude models)."""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from agent_harness.core.config import LLMConfig
from agent_harness.core.errors import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMRateLimitError,
)
from agent_harness.core.message import Message, MessageChunk, Role, ToolCall
from agent_harness.llm.base import BaseLLM
from agent_harness.llm.types import FinishReason, LLMResponse, StreamDelta, Usage
from agent_harness.tool.base import ToolSchema

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLM):
    """Anthropic API provider (Claude 3.5, Claude 4, etc.).

    Handles Anthropic-specific differences:
    - System message is a top-level parameter, not in messages
    - Tool results use content blocks, not separate message role
    - Streaming uses server-sent events
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        request_kwargs = self._build_request(
            messages, tools, tool_choice, temperature, max_tokens, **kwargs
        )

        try:
            response = await self._client.messages.create(**request_kwargs)
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(str(e)) from e
        except anthropic.BadRequestError as e:
            if "context" in str(e).lower() or "too long" in str(e).lower():
                raise LLMContextLengthError(str(e)) from e
            raise LLMError(str(e)) from e
        except anthropic.APIError as e:
            raise LLMError(str(e)) from e

        return self._parse_response(response)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamDelta]:
        request_kwargs = self._build_request(
            messages, tools, tool_choice, temperature, max_tokens, **kwargs
        )

        try:
            async with self._client.messages.stream(**request_kwargs) as stream:
                async for event in stream:
                    delta = self._parse_stream_event(event)
                    if delta:
                        yield delta
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(str(e)) from e
        except anthropic.APIError as e:
            raise LLMError(str(e)) from e

    def _build_request(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        tool_choice: str | None,
        temperature: float | None,
        max_tokens: int | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Extract system message (Anthropic takes it as top-level param)
        system_content, api_messages = self._split_system_message(messages)

        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": api_messages,
            "temperature": self._resolve_temperature(temperature),
            "max_tokens": self._resolve_max_tokens(max_tokens),
        }

        if system_content:
            request["system"] = system_content

        if tools:
            request["tools"] = [t.to_anthropic_format() for t in tools]
            if tool_choice:
                if tool_choice == "auto":
                    request["tool_choice"] = {"type": "auto"}
                elif tool_choice == "required":
                    request["tool_choice"] = {"type": "any"}
                elif tool_choice == "none":
                    pass  # Don't send tools
                else:
                    request["tool_choice"] = {"type": "tool", "name": tool_choice}

        request.update(kwargs)
        return request

    @staticmethod
    def _split_system_message(
        messages: list[Message],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Separate system message and format remaining for Anthropic API."""
        system_content: str | None = None
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Anthropic: system is a top-level parameter
                system_content = msg.content
                continue

            if msg.role == Role.TOOL and msg.tool_result:
                # Anthropic: tool results are user messages with tool_result content blocks
                api_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_result.tool_call_id,
                            "content": msg.tool_result.content,
                            "is_error": msg.tool_result.is_error,
                        }
                    ],
                })
                continue

            if msg.role == Role.ASSISTANT and msg.tool_calls:
                # Anthropic: tool calls are content blocks in assistant message
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                api_messages.append({
                    "role": "assistant",
                    "content": content_blocks,
                })
                continue

            # Regular message
            api_messages.append({
                "role": msg.role.value,
                "content": msg.content or "",
            })

        return system_content, api_messages

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        """Convert Anthropic response to LLMResponse."""
        content_text: str | None = None
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        message = Message(
            role=Role.ASSISTANT,
            content=content_text,
            tool_calls=tool_calls if tool_calls else None,
        )

        usage = Usage(
            prompt_tokens=response.usage.input_tokens if response.usage else 0,
            completion_tokens=response.usage.output_tokens if response.usage else 0,
            total_tokens=(
                (response.usage.input_tokens + response.usage.output_tokens)
                if response.usage else 0
            ),
        )

        finish_reason = FinishReason.STOP
        if response.stop_reason == "tool_use":
            finish_reason = FinishReason.TOOL_CALLS
        elif response.stop_reason == "max_tokens":
            finish_reason = FinishReason.LENGTH

        return LLMResponse(
            message=message,
            usage=usage,
            finish_reason=finish_reason,
            model=response.model,
            raw_response=response,
        )

    @staticmethod
    def _parse_stream_event(event: Any) -> StreamDelta | None:
        """Parse an Anthropic stream event into a StreamDelta."""
        # Handle different event types from Anthropic's streaming API
        event_type = getattr(event, "type", None)

        if event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and getattr(delta, "type", None) == "text_delta":
                return StreamDelta(
                    chunk=MessageChunk(delta_content=delta.text),
                )
            if delta and getattr(delta, "type", None) == "input_json_delta":
                # Partial tool input JSON
                return None  # Accumulate externally

        if event_type == "message_delta":
            stop_reason = getattr(getattr(event, "delta", None), "stop_reason", None)
            if stop_reason:
                finish = FinishReason.STOP
                if stop_reason == "tool_use":
                    finish = FinishReason.TOOL_CALLS
                elif stop_reason == "max_tokens":
                    finish = FinishReason.LENGTH
                return StreamDelta(
                    chunk=MessageChunk(finish_reason=stop_reason),
                    finish_reason=finish,
                )

        return None
