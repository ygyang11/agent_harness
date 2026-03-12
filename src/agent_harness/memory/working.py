"""Working memory: structured scratchpad for agent reasoning state."""
from __future__ import annotations

from typing import Any

from agent_harness.core.message import Message
from agent_harness.memory.base import BaseMemory, MemoryItem
from agent_harness.memory.retrieval import HybridRetriever


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
        """Serialize the scratchpad to a structured string for prompt injection.

        Produces a structured Markdown output that helps the LLM understand
        the current agent state, including goal, plan progress, current step,
        observations, errors, and other variables.
        """
        if not self._scratchpad and not self._history:
            return ""

        sections: list[str] = ["## Working Memory"]

        # Known structured keys rendered in order
        _structured_keys = {"goal", "plan", "current_step", "errors"}

        if goal := self._scratchpad.get("goal"):
            sections.append(f"\n### Current Goal\n{goal}")

        if plan := self._scratchpad.get("plan"):
            plan_str = str(plan)
            if len(plan_str) > 1000:
                plan_str = plan_str[:1000] + "..."
            sections.append(f"\n### Plan Progress\n{plan_str}")

        if step := self._scratchpad.get("current_step"):
            sections.append(f"\n### Current Step\n{step}")

        if self._history:
            recent = self._history[-10:]
            obs_lines = [f"- {entry[:200]}" for entry in recent]
            sections.append("\n### Key Observations\n" + "\n".join(obs_lines))

        if errors := self._scratchpad.get("errors"):
            errors_str = str(errors)
            if len(errors_str) > 500:
                errors_str = errors_str[:500] + "..."
            sections.append(f"\n### Errors & Retries\n{errors_str}")

        # Remaining variables
        other_keys = [
            k for k in self._scratchpad if k not in _structured_keys
        ]
        if other_keys:
            var_lines: list[str] = []
            for key in other_keys:
                val_str = str(self._scratchpad[key])
                if len(val_str) > 500:
                    val_str = val_str[:500] + "..."
                var_lines.append(f"- **{key}**: {val_str}")
            sections.append("\n### Variables\n" + "\n".join(var_lines))

        return "\n".join(sections)

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
        """Query history entries using hybrid retrieval."""
        if not self._history:
            return []
        items = [MemoryItem(content=entry) for entry in self._history]
        retriever = HybridRetriever()
        return retriever.retrieve(query, items, top_k=top_k)

    async def get_context_messages(self) -> list[Message]:
        """Inject working memory as a system message."""
        prompt_str = self.to_prompt_string()
        if not prompt_str:
            return []
        return [Message.system(prompt_str)]

    async def clear(self) -> None:
        self._scratchpad.clear()
        self._history.clear()

    async def forget(self, threshold: float = 0.3) -> int:
        """Remove old history entries (scratchpad keys are preserved)."""
        if not self._history:
            return 0
        # Keep the most recent entries, drop old ones beyond a reasonable cap
        max_keep = max(10, int(len(self._history) * threshold))
        removed = max(0, len(self._history) - max_keep)
        if removed > 0:
            self._history = self._history[-max_keep:]
        return removed

    async def size(self) -> int:
        return len(self._scratchpad) + len(self._history)
