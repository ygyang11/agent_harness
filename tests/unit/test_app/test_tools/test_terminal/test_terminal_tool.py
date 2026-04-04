"""Tests for the refactored terminal_tool."""

from __future__ import annotations

import pytest

from agent_app.tools.terminal.terminal_tool import terminal_tool
from agent_harness.core.config import ToolConfig
from agent_harness.core.message import ToolCall
from agent_harness.tool.decorator import tool
from agent_harness.tool.executor import ToolExecutor
from agent_harness.tool.registry import ToolRegistry


class TestTerminalTool:
    """Unit tests — direct tool execution."""

    # --- Basic execution ---

    @pytest.mark.asyncio
    async def test_simple_command(self) -> None:
        result = await terminal_tool.execute(command="echo hello")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_empty_command(self) -> None:
        result = await terminal_tool.execute(command="")
        assert result == "Error: command cannot be empty"

    @pytest.mark.asyncio
    async def test_whitespace_only_command(self) -> None:
        result = await terminal_tool.execute(command="   ")
        assert result == "Error: command cannot be empty"

    @pytest.mark.asyncio
    async def test_no_output(self) -> None:
        result = await terminal_tool.execute(command="true")
        assert result == "(no output)"

    @pytest.mark.asyncio
    async def test_non_zero_exit(self) -> None:
        result = await terminal_tool.execute(
            command='python3 -c "import sys; sys.exit(2)"', timeout=5,
        )
        assert "[exit code 2]" in result

    @pytest.mark.asyncio
    async def test_stderr_merged(self) -> None:
        result = await terminal_tool.execute(command="echo err >&2 && echo out", timeout=5)
        assert "out" in result
        assert "err" in result

    # --- Full bash syntax (all blocked by old implementation, now allowed) ---

    @pytest.mark.asyncio
    async def test_pipe(self) -> None:
        result = await terminal_tool.execute(command="echo hello | tr 'h' 'H'", timeout=5)
        assert "Hello" in result

    @pytest.mark.asyncio
    async def test_variable_expansion(self) -> None:
        result = await terminal_tool.execute(command="X=42 && echo $X", timeout=5)
        assert "42" in result

    @pytest.mark.asyncio
    async def test_git(self) -> None:
        result = await terminal_tool.execute(command="git --version", timeout=5)
        assert "git version" in result

    @pytest.mark.asyncio
    async def test_subshell(self) -> None:
        result = await terminal_tool.execute(command="echo $(echo nested)", timeout=5)
        assert "nested" in result

    @pytest.mark.asyncio
    async def test_redirect(self) -> None:
        result = await terminal_tool.execute(
            command="echo data > out.txt && cat out.txt", timeout=5,
        )
        assert "data" in result

    # --- Timeout ---

    @pytest.mark.asyncio
    async def test_timeout_enforced(self) -> None:
        result = await terminal_tool.execute(command="sleep 10", timeout=1)
        assert "timed out" in result

    @pytest.mark.asyncio
    async def test_timeout_capped_at_max(self) -> None:
        result = await terminal_tool.execute(command="echo ok", timeout=1000)
        assert "ok" in result

    @pytest.mark.asyncio
    async def test_timeout_zero(self) -> None:
        result = await terminal_tool.execute(command="echo x", timeout=0)
        assert result == "Error: timeout must be greater than 0"

    @pytest.mark.asyncio
    async def test_timeout_negative(self) -> None:
        result = await terminal_tool.execute(command="echo x", timeout=-1)
        assert result == "Error: timeout must be greater than 0"

    # --- Output truncation ---

    @pytest.mark.asyncio
    async def test_large_output_truncated(self) -> None:
        result = await terminal_tool.execute(
            command='python3 -c "print(\'x \' * 50000)"', timeout=10,
        )
        assert "... (truncated)" in result


class TestTerminalToolIntegration:
    """Integration tests through ToolExecutor — validates timeout chain."""

    @pytest.mark.asyncio
    async def test_executor_timeout_overrides_short_default(self) -> None:
        """executor_timeout (610s) overrides a very short default_timeout (1s)."""
        registry = ToolRegistry()
        registry.register(terminal_tool)
        config = ToolConfig(default_timeout=1.0)
        executor = ToolExecutor(registry, config=config)

        tc = ToolCall(
            name="terminal_tool",
            arguments={"command": "sleep 2 && echo done", "timeout": 5},
        )
        result = await executor.execute(tc)
        assert not result.is_error
        assert "done" in result.content

    @pytest.mark.asyncio
    async def test_internal_timeout_fires_before_executor(self) -> None:
        """Terminal's internal timeout (1s) fires, not executor's outer timeout."""
        registry = ToolRegistry()
        registry.register(terminal_tool)
        executor = ToolExecutor(registry)

        tc = ToolCall(
            name="terminal_tool",
            arguments={"command": "sleep 60", "timeout": 1},
        )
        result = await executor.execute(tc)
        assert "timed out after 1s" in result.content
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_executor_timeout_attribute(self) -> None:
        """terminal_tool declares executor_timeout > 600; default tools don't."""
        assert terminal_tool.executor_timeout is not None
        assert terminal_tool.executor_timeout > 600

        @tool
        async def my_tool(x: str) -> str:
            return x

        assert my_tool.executor_timeout is None
