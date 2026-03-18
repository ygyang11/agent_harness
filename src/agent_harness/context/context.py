"""Unified agent context container."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from agent_harness.core.config import HarnessConfig
from agent_harness.core.event import EventBus
from agent_harness.core.message import Message, Role
from agent_harness.memory.short_term import ShortTermMemory
from agent_harness.memory.working_term import WorkingMemory
from agent_harness.context.state import StateManager
from agent_harness.context.variables import ContextVariables

if TYPE_CHECKING:
    from agent_harness.memory.long_term import LongTermMemory
    from agent_harness.tracing.tracer import Tracer

logger = logging.getLogger(__name__)


class AgentContext:
    """Unified context for agent execution.

    Aggregates all runtime dependencies an agent needs:
    memory (short/long/working), state machine, shared variables,
    config, event bus, and tracer.

    Use fork() to create child contexts for sub-agents that share
    long-term memory and global variables but have independent
    short-term memory and state.
    """

    def __init__(
        self,
        config: HarnessConfig | None = None,
        short_term_memory: ShortTermMemory | None = None,
        long_term_memory: LongTermMemory | None = None,
        working_memory: WorkingMemory | None = None,
        state: StateManager | None = None,
        variables: ContextVariables | None = None,
        event_bus: EventBus | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.config = config or HarnessConfig.get()
        self.short_term_memory = short_term_memory or ShortTermMemory(
            max_messages=self.config.memory.short_term_max_messages,
            max_tokens=self.config.memory.short_term_max_tokens,
            strategy=self.config.memory.short_term_strategy,
            model=self.config.llm.model,
        )
        self.long_term_memory = long_term_memory
        self.working_memory = working_memory or WorkingMemory()
        self.state = state or StateManager()
        self.variables = variables or ContextVariables()
        self.event_bus = event_bus or EventBus()
        self.tracer = tracer

    @classmethod
    def create(cls, config: HarnessConfig | None = None, **kwargs: Any) -> AgentContext:
        """Factory method to create a context with default components."""
        return cls(config=config, **kwargs)

    def fork(self, name: str | None = None) -> AgentContext:
        """Create a child context for a sub-agent.

        Child context shares:
        - long_term_memory (shared knowledge base)
        - variables (GLOBAL scope variables)
        - event_bus (unified event stream)
        - tracer (unified tracing)
        - config

        Child gets fresh:
        - short_term_memory (independent conversation)
        - working_memory (independent scratchpad)
        - state (independent state machine)
        """
        child_vars = self.variables.fork()
        return AgentContext(
            config=self.config,
            short_term_memory=ShortTermMemory(
                max_messages=self.config.memory.short_term_max_messages,
                max_tokens=self.config.memory.short_term_max_tokens,
                strategy=self.config.memory.short_term_strategy,
                model=self.config.llm.model,
            ),
            long_term_memory=self.long_term_memory,
            working_memory=WorkingMemory(),
            state=StateManager(),
            variables=child_vars,
            event_bus=self.event_bus,
            tracer=self.tracer,
        )

    def __repr__(self) -> str:
        return (
            f"AgentContext(state={self.state.current}, "
            f"has_long_term={self.long_term_memory is not None})"
        )

    async def build_llm_messages(
        self,
        base_messages: list[Message] | None = None,
        *,
        include_working: bool = True,
        include_long_term: bool = False,
        long_term_query: str | None = None,
        long_term_top_k: int = 3,
    ) -> list[Message]:
        """Build the full message list for an LLM call.

        Assembles messages from short-term memory, injects working memory
        context, and optionally retrieves relevant long-term memories.

        Args:
            base_messages: Pre-built messages. If None, reads from short_term_memory.
            include_working: Inject working memory after system message.
            include_long_term: Query long_term_memory and inject results.
            long_term_query: Query text for long-term retrieval. Falls back
                to the last user message content.
            long_term_top_k: Max items to retrieve from long-term memory.

        Returns:
            Ordered list of messages ready for LLM consumption.
        """
        messages = base_messages
        if messages is None:
            messages = await self.short_term_memory.get_context_messages()

        # Determine injection point (after system message if present)
        inject_idx = 1 if messages and messages[0].role == Role.SYSTEM else 0

        # Working memory injection
        if include_working:
            working_msgs = await self.working_memory.get_context_messages()
            if working_msgs:
                messages = messages[:inject_idx] + working_msgs + messages[inject_idx:]
                inject_idx += len(working_msgs)

        # Long-term memory injection
        if include_long_term and self.long_term_memory is not None:
            query = long_term_query
            if not query:
                # Fall back to last user message content
                for msg in reversed(messages):
                    if msg.role == Role.USER and msg.content:
                        query = msg.content
                        break

            if query:
                try:
                    items = await self.long_term_memory.query(query, top_k=long_term_top_k)
                    if items:
                        context_parts = [item.content for item in items]
                        lt_msg = Message.system(
                            "## Relevant Knowledge\n\n" + "\n\n---\n\n".join(context_parts)
                        )
                        messages = messages[:inject_idx] + [lt_msg] + messages[inject_idx:]
                except Exception as e:
                    logger.warning("Failed to query long-term memory: %s", e)

        return messages
