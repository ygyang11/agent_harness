"""Working memory: structured scratchpad for agent reasoning state."""
from __future__ import annotations

from typing import Any

from agent_harness.core.message import Message
from agent_harness.memory.base import BaseMemory, MemoryItem


class WorkingMemory(BaseMemory):
    """Structured scratchpad for storing intermediate reasoning state.

    Used by agents to maintain structured data during execution:
    - PlanAgent stores the current plan and step status
    - ReActAgent stores the current thought chain
    - Research agents store notes, findings, and intermediate results

    Data is stored as key-value pairs and can be serialized to
    a prompt-injectable string.
    """

    def __init__(self) -> None:
        self._scratchpad: dict[str, Any] = {}
        self._history: list[str] = []  # ordered log of additions

    # --- Key-value interface ---

    def set(self, key: str, value: Any) -> None:
        """Set a key-value pair in the scratchpad."""
        self._scratchpad[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the scratchpad."""
        return self._scratchpad.get(key, default)

    def delete(self, key: str) -> None:
        """Delete a key from the scratchpad."""
        self._scratchpad.pop(key, None)

    def keys(self) -> list[str]:
        """List all keys in the scratchpad."""
        return list(self._scratchpad.keys())

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the scratchpad as a dict."""
        return dict(self._scratchpad)

    # --- Prompt serialization ---

    def to_prompt_string(self) -> str:
        """Serialize the scratchpad to a string for prompt injection.

        Format:
            [Working Memory]
            key1: value1
            key2: value2
        """
        if not self._scratchpad:
            return ""

        lines = ["[Working Memory]"]
        for key, value in self._scratchpad.items():
            val_str = str(value)
            if len(val_str) > 500:
                val_str = val_str[:500] + "..."
            lines.append(f"  {key}: {val_str}")
        return "\n".join(lines)

    # --- BaseMemory interface ---

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a note to the working memory history."""
        self._history.append(content)
        if metadata:
            for key, value in metadata.items():
                self._scratchpad[key] = value

    async def add_message(self, message: Message) -> None:
        """Add a message's content to history."""
        if message.content:
            self._history.append(message.content)

    async def query(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        """Return the most recent history entries."""
        recent = self._history[-top_k:]
        return [MemoryItem(content=item) for item in recent]

    async def get_context_messages(self) -> list[Message]:
        """Inject working memory as a system message."""
        prompt_str = self.to_prompt_string()
        if not prompt_str:
            return []
        return [Message.system(prompt_str)]

    async def clear(self) -> None:
        self._scratchpad.clear()
        self._history.clear()

    async def size(self) -> int:
        return len(self._scratchpad) + len(self._history)
