"""OpenAI LLM provider for agent_harness."""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import openai
from openai import AsyncOpenAI

from agent_harness.core.config import HarnessConfig, LLMConfig, resolve_llm_config
from agent_harness.core.errors import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
)
from agent_harness.core.message import Message, MessageChunk, Role, ToolCall
from agent_harness.llm.base import BaseLLM
from agent_harness.llm.types import FinishReason, LLMResponse, StreamDelta, Usage
from agent_harness.tool.base import ToolSchema

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLM):
    """OpenAI API provider (GPT-4o, o1, etc.).

    Handles:
    - Message format conversion (Message -> OpenAI dict)
    - Function/tool calling
    - Streaming
    - Error mapping to framework exceptions
    """

    def __init__(self, config: HarnessConfig | LLMConfig | None = None) -> None:
        llm_config = resolve_llm_config(config)
        super().__init__(llm_config)
        self._client = AsyncOpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            timeout=llm_config.timeout,
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
            response = await self._client.chat.completions.create(**request_kwargs)
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except openai.AuthenticationError as e:
            raise LLMAuthenticationError(str(e)) from e
        except openai.BadRequestError as e:
            if "context_length" in str(e).lower() or "maximum context" in str(e).lower():
                raise LLMContextLengthError(str(e)) from e
            raise LLMError(str(e)) from e
        except openai.APIError as e:
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
            messages, tools, tool_choice, temperature, max_tokens, stream=True, **kwargs
        )

        try:
            stream = await self._client.chat.completions.create(**request_kwargs)
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except openai.AuthenticationError as e:
            raise LLMAuthenticationError(str(e)) from e
        except openai.APIError as e:
            raise LLMError(str(e)) from e

        tc_buffer: dict[int, dict[str, str]] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tc_buffer:
                        tc_buffer[idx] = {"id": "", "name": "", "args": ""}
                    if tc.id:
                        tc_buffer[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tc_buffer[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tc_buffer[idx]["args"] += tc.function.arguments

            finish_reason = None
            delta_tool_calls = None
            if choice.finish_reason:
                finish_reason = _map_finish_reason(choice.finish_reason)
                if tc_buffer and finish_reason == FinishReason.TOOL_CALLS:
                    delta_tool_calls = []
                    for idx in sorted(tc_buffer):
                        buf = tc_buffer[idx]
                        try:
                            args = json.loads(buf["args"]) if buf["args"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        delta_tool_calls.append(
                            ToolCall(id=buf["id"], name=buf["name"], arguments=args)
                        )

            yield StreamDelta(
                chunk=MessageChunk(
                    delta_content=delta.content,
                    delta_tool_calls=delta_tool_calls,
                    finish_reason=choice.finish_reason,
                ),
                finish_reason=finish_reason,
            )

    def _build_request(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        tool_choice: str | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [self._format_message(m) for m in messages],
            "temperature": self._resolve_temperature(temperature),
            "max_tokens": self._resolve_max_tokens(max_tokens),
            "stream": stream,
        }

        if tools:
            request["tools"] = [t.to_openai_format() for t in tools]
            if tool_choice:
                if tool_choice in ("auto", "required", "none"):
                    request["tool_choice"] = tool_choice
                else:
                    request["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }

        if self.config.reasoning_effort:
            request["reasoning_effort"] = self.config.reasoning_effort

        request.update(kwargs)
        return request

    @staticmethod
    def _format_message(msg: Message) -> dict[str, Any]:
        """Convert a Message to OpenAI API format."""
        result: dict[str, Any] = {"role": msg.role.value}

        if msg.role == Role.TOOL and msg.tool_result:
            result["tool_call_id"] = msg.tool_result.tool_call_id
            result["content"] = msg.tool_result.content
            return result

        if msg.content is not None:
            result["content"] = msg.content

        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]

        if msg.name:
            result["name"] = msg.name

        return result

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        """Convert OpenAI response to LLMResponse."""
        if not response.choices:
            raise LLMResponseError(
                "OpenAI API returned empty choices",
                details={"model": response.model},
            )
        choice = response.choices[0]
        msg = choice.message

        # Parse tool calls
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments) if tc.function.arguments else {},
                )
                for tc in msg.tool_calls
            ]

        message = Message(
            role=Role.ASSISTANT,
            content=msg.content,
            tool_calls=tool_calls,
        )

        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )

        return LLMResponse(
            message=message,
            usage=usage,
            finish_reason=_map_finish_reason(choice.finish_reason),
            model=response.model,
            raw_response=response,
        )


def _map_finish_reason(reason: str | None) -> FinishReason:
    """Map OpenAI finish reason to FinishReason enum."""
    mapping = {
        "stop": FinishReason.STOP,
        "tool_calls": FinishReason.TOOL_CALLS,
        "length": FinishReason.LENGTH,
        "content_filter": FinishReason.CONTENT_FILTER,
    }
    return mapping.get(reason or "stop", FinishReason.STOP)
