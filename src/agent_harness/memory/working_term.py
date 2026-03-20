"""Working memory: structured scratchpad for agent reasoning state."""
from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Any

from agent_harness.core.message import Message
from agent_harness.memory.base import BaseMemory, MemoryItem
from agent_harness.memory.retrieval import HybridRetriever
from agent_harness.utils.token_counter import truncate_text_by_tokens


class ScratchpadCategory(str, Enum):
    """Semantic categories for scratchpad keys."""
    OBJECTIVE = "objective"
    PLAN = "plan"
    PROGRESS = "progress"
    OBSERVATION = "observation"
    ERROR = "error"


_DEFAULT_KEY_CATEGORIES: dict[str, ScratchpadCategory] = {
    "goal": ScratchpadCategory.OBJECTIVE,
    "task": ScratchpadCategory.OBJECTIVE,
    "objective": ScratchpadCategory.OBJECTIVE,
    "mission": ScratchpadCategory.OBJECTIVE,
    "plan": ScratchpadCategory.PLAN,
    "steps": ScratchpadCategory.PLAN,
    "strategy": ScratchpadCategory.PLAN,
    "roadmap": ScratchpadCategory.PLAN,
    "current_step": ScratchpadCategory.PROGRESS,
    "current_task": ScratchpadCategory.PROGRESS,
    "progress": ScratchpadCategory.PROGRESS,
    "status": ScratchpadCategory.PROGRESS,
    "observations": ScratchpadCategory.OBSERVATION,
    "findings": ScratchpadCategory.OBSERVATION,
    "notes": ScratchpadCategory.OBSERVATION,
    "errors": ScratchpadCategory.ERROR,
    "failures": ScratchpadCategory.ERROR,
    "retries": ScratchpadCategory.ERROR,
}

_CATEGORY_HEADINGS: dict[ScratchpadCategory, str] = {
    ScratchpadCategory.OBJECTIVE: "Objective",
    ScratchpadCategory.PLAN: "Plan",
    ScratchpadCategory.PROGRESS: "Current Progress",
    ScratchpadCategory.OBSERVATION: "Observations",
    ScratchpadCategory.ERROR: "Errors",
}

_MAX_VALUE_TOKENS = 5000


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
        self._history: list[Message] = []
        self._key_categories: dict[str, ScratchpadCategory] = {}

    # --- Key-value interface ---

    def set(self, key: str, value: Any, *, category: str | None = None) -> None:
        """Set a scratchpad key-value pair with optional category."""
        self._scratchpad[key] = value
        if category is not None:
            self._key_categories[key] = ScratchpadCategory(category)

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

    def _get_category(self, key: str) -> ScratchpadCategory | None:
        """Resolve the category for a scratchpad key."""
        if key in self._key_categories:
            return self._key_categories[key]
        return _DEFAULT_KEY_CATEGORIES.get(key.lower())

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a value for prompt rendering, truncating if too long."""
        text = str(value)
        return truncate_text_by_tokens(
            text,
            max_tokens=_MAX_VALUE_TOKENS,
            suffix="...",
        )

    def to_prompt_string(self) -> str:
        """Render scratchpad and history as a structured prompt string."""
        if not self._scratchpad and not self._history:
            return ""

        sections: list[str] = []

        # Group keys by category
        categorized: dict[ScratchpadCategory, list[tuple[str, Any]]] = {}
        uncategorized: list[tuple[str, Any]] = []

        for key, value in self._scratchpad.items():
            cat = self._get_category(key)
            if cat is not None:
                categorized.setdefault(cat, []).append((key, value))
            else:
                uncategorized.append((key, value))

        # Render categorized sections in enum order
        for cat in ScratchpadCategory:
            if cat in categorized:
                heading = _CATEGORY_HEADINGS[cat]
                lines = [f"### {heading}"]
                for key, value in categorized[cat]:
                    formatted = self._format_value(value)
                    lines.append(f"**{key}**: \n{formatted}")
                sections.append("\n".join(lines))

        # Render uncategorized keys
        if uncategorized:
            lines = ["### Additional Context"]
            for key, value in uncategorized:
                formatted = self._format_value(value)
                lines.append(f"**{key}**: \n{formatted}")
            sections.append("\n".join(lines))

        # Render history
        if self._history:
            lines = ["### Key Observations"]
            for msg in self._history:
                lines.append(f"- {msg.content}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    # --- BaseMemory interface ---

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a note to the working memory history."""
        meta = dict(metadata) if metadata else {}
        if "importance_score" not in meta:
            meta["importance_score"] = 0.5
        msg = Message.user(content, metadata=meta)
        self._history.append(msg)
        if metadata:
            for key, value in metadata.items():
                if key != "importance_score":
                    self._scratchpad[key] = value

    async def add_message(self, message: Message) -> None:
        """Add a message to history."""
        if message.content:
            self._history.append(message)

    async def query(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        """Query history entries using hybrid retrieval."""
        if not self._history:
            return []
        items = [
            MemoryItem(
                content=msg.content or "",
                metadata={"role": msg.role.value},
                importance_score=msg.metadata.get("importance_score", 0.5),
                timestamp=msg.created_at,
            )
            for msg in self._history
            if msg.content
        ]
        if not items:
            return []
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
        """Remove history entries below the importance-weighted time-decay threshold.

        Scratchpad keys are always preserved.
        """
        now = datetime.now()
        original = len(self._history)
        kept: list[Message] = []
        for msg in self._history:
            hours = (now - msg.created_at).total_seconds() / 3600
            time_decay = math.exp(-0.01 * hours)
            importance = msg.metadata.get("importance_score", 0.5)
            weighted = time_decay * (0.8 + importance * 0.4)
            if weighted >= threshold:
                kept.append(msg)
        self._history = kept
        return original - len(kept)

    async def size(self) -> int:
        return len(self._scratchpad) + len(self._history)
