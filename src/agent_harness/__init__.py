"""Agent Harness: A complete, extensible agent framework."""
from agent_harness.utils.logging_config import setup_logging

# Auto-configure logging on import
setup_logging()

from agent_harness.core.message import Message, Role, ToolCall, ToolResult
from agent_harness.core.config import (
    HarnessConfig,
    LLMConfig,
    MemoryConfig,
    PaperConfig,
    SearchConfig,
    SkillConfig,
    ToolConfig,
    TracingConfig,
)
from agent_harness.skills.loader import Skill, SkillLoader
from agent_harness.core.event import Event, EventBus
from agent_harness.core.errors import HarnessError
from agent_harness.llm.base import BaseLLM
from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.tool.decorator import tool
from agent_harness.agent.base import BaseAgent, AgentResult
from agent_harness.agent.react import ReActAgent
from agent_harness.agent.planner import PlanAgent, PlanAndExecuteAgent
from agent_harness.agent.conversational import ConversationalAgent
from agent_harness.context.context import AgentContext

__version__ = "0.2.0"

__all__ = [
    # Logging
    "setup_logging",
    # Core
    "Message",
    "Role",
    "ToolCall",
    "ToolResult",
    "HarnessConfig",
    "LLMConfig",
    "MemoryConfig",
    "TracingConfig",
    "ToolConfig",
    "SearchConfig",
    "PaperConfig",
    "SkillConfig",
    "Event",
    "EventBus",
    "HarnessError",
    # LLM
    "BaseLLM",
    # Tool
    "BaseTool",
    "ToolSchema",
    "tool",
    # Agent
    "BaseAgent",
    "AgentResult",
    "ReActAgent",
    "PlanAgent",
    "PlanAndExecuteAgent",
    "ConversationalAgent",
    # Context
    "AgentContext",
    # Skills
    "Skill",
    "SkillLoader",
]
