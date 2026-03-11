"""Fluent prompt builder for constructing message sequences."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from agent_harness.core.message import Message, Role

if TYPE_CHECKING:
    from agent_harness.memory.base import BaseMemory
    from agent_harness.prompt.template import PromptTemplate


class PromptBuilder:
    """Fluent API for building message sequences.

    Example:
        messages = (
            PromptBuilder()
            .system("You are a helpful assistant.")
            .user("What is AI?")
            .build()
        )
    """

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def system(self, content: str, **metadata: Any) -> PromptBuilder:
        """Add a system message."""
        self._messages.append(Message.system(content, metadata=metadata))
        return self

    def user(self, content: str, **metadata: Any) -> PromptBuilder:
        """Add a user message."""
        self._messages.append(Message.user(content, metadata=metadata))
        return self

    def assistant(self, content: str, **metadata: Any) -> PromptBuilder:
        """Add an assistant message."""
        self._messages.append(Message.assistant(content, metadata=metadata))
        return self

    def message(self, msg: Message) -> PromptBuilder:
        """Add a pre-built message."""
        self._messages.append(msg)
        return self

    def messages(self, msgs: list[Message]) -> PromptBuilder:
        """Add multiple pre-built messages."""
        self._messages.extend(msgs)
        return self

    def template(self, tmpl: PromptTemplate, role: str = "system", **kwargs: Any) -> PromptBuilder:
        """Render a PromptTemplate and add as a message."""
        content = tmpl.render(**kwargs)
        self._messages.append(Message(role=Role(role), content=content))
        return self

    async def from_memory(self, memory: BaseMemory) -> PromptBuilder:
        """Add messages from a memory instance."""
        mem_messages = await memory.get_context_messages()
        self._messages.extend(mem_messages)
        return self

    def build(self) -> list[Message]:
        """Build and return the message list."""
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)
