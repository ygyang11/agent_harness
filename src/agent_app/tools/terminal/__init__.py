"""Terminal tool — shell command execution for agent applications."""

from agent_app.tools.terminal.terminal_tool import terminal_tool

TERMINAL_TOOLS = [terminal_tool]

__all__ = ["terminal_tool", "TERMINAL_TOOLS"]
