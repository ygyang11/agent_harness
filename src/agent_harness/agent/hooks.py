"""Agent lifecycle hooks for extensible behavior."""
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_harness.core.message import Message, ToolCall, ToolResult


@runtime_checkable
class AgentHooks(Protocol):
    """Protocol defining agent lifecycle hooks.
    
    Implement this to observe or modify agent behavior at key points.
    All methods are optional — provide a default no-op base class.
    """

    async def on_run_start(self, agent_name: str, input_text: str) -> None: ...
    async def on_step_start(self, agent_name: str, step: int) -> None: ...
    async def on_llm_call(self, agent_name: str, messages: list[Message]) -> None: ...
    async def on_tool_call(self, agent_name: str, tool_call: ToolCall) -> None: ...
    async def on_tool_result(self, agent_name: str, result: ToolResult) -> None: ...
    async def on_step_end(self, agent_name: str, step: int) -> None: ...
    async def on_run_end(self, agent_name: str, output: str) -> None: ...
    async def on_error(self, agent_name: str, error: Exception) -> None: ...


class DefaultHooks:
    """No-op implementation of AgentHooks. Subclass to override specific hooks."""

    async def on_run_start(self, agent_name: str, input_text: str) -> None:
        pass

    async def on_step_start(self, agent_name: str, step: int) -> None:
        pass

    async def on_llm_call(self, agent_name: str, messages: list[Any]) -> None:
        pass

    async def on_tool_call(self, agent_name: str, tool_call: Any) -> None:
        pass

    async def on_tool_result(self, agent_name: str, result: Any) -> None:
        pass

    async def on_step_end(self, agent_name: str, step: int) -> None:
        pass

    async def on_run_end(self, agent_name: str, output: str) -> None:
        pass

    async def on_error(self, agent_name: str, error: Exception) -> None:
        pass
