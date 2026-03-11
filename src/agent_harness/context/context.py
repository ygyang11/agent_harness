"""Unified agent context container."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from agent_harness.core.config import HarnessConfig
from agent_harness.core.event import EventBus
from agent_harness.memory.short_term import ShortTermMemory
from agent_harness.memory.working import WorkingMemory
from agent_harness.context.state import StateManager
from agent_harness.context.variables import ContextVariables

if TYPE_CHECKING:
    from agent_harness.memory.long_term import LongTermMemory
    from agent_harness.tracing.tracer import Tracer


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
        self.config = config or HarnessConfig()
        self.short_term_memory = short_term_memory or ShortTermMemory(
            max_messages=self.config.memory.short_term_max_messages,
            max_tokens=self.config.memory.short_term_max_tokens,
            strategy=self.config.memory.short_term_strategy,
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
