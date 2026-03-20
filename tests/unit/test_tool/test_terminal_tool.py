"""Tests for unified terminal tool."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness.tool.builtin.terminal_tool import terminal_tool


class TestTerminalTool:
    @pytest.mark.asyncio
    async def test_empty_command_returns_error(self) -> None:
        result = await terminal_tool.execute(command="")
        assert result == "Error: command cannot be empty"

    @pytest.mark.asyncio
    async def test_timeout_must_be_positive(self) -> None:
        result = await terminal_tool.execute(command="pwd", timeout=0)
        assert result == "Error: timeout must be greater than 0"

    @pytest.mark.asyncio
    async def test_pwd_and_ls_chain_succeeds(self) -> None:
        result = await terminal_tool.execute(command="pwd && ls", timeout=5)
        assert "agent_harness" in result

    @pytest.mark.asyncio
    async def test_chain_with_disallowed_subcommand_is_blocked(self) -> None:
        result = await terminal_tool.execute(command="pwd && whoami", timeout=5)
        assert result.startswith("Error:")
        assert "command not allowed" in result

    @pytest.mark.asyncio
    async def test_chain_with_dangerous_token_is_blocked(self) -> None:
        result = await terminal_tool.execute(command="ls ; rm -rf /tmp/demo", timeout=5)
        assert result.startswith("Error:")
        assert "disallowed token detected" in result

    @pytest.mark.asyncio
    async def test_bash_executor_is_blocked(self) -> None:
        result = await terminal_tool.execute(command="bash script.sh", timeout=5)
        assert result.startswith("Error:")
        assert "blocked command executor" in result

    @pytest.mark.asyncio
    async def test_cwd_escape_is_blocked(self) -> None:
        result = await terminal_tool.execute(command="pwd", cwd="../..", timeout=5)
        assert result.startswith("Error:")
        assert "path escapes workspace" in result

    @pytest.mark.asyncio
    async def test_python_c_runs_successfully(self) -> None:
        result = await terminal_tool.execute(command='python -c "print(123)"', timeout=5)
        assert "123" in result

    @pytest.mark.asyncio
    async def test_non_zero_exit_contains_exit_code(self) -> None:
        result = await terminal_tool.execute(command='python -c "import sys; sys.exit(2)"', timeout=5)
        assert result.startswith("[exit code 2]")

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self) -> None:
        result = await terminal_tool.execute(
            command='python -c "import time; time.sleep(2)"',
            timeout=1,
        )
        assert result == "Error: execution timed out after 1s"

    @pytest.mark.asyncio
    async def test_output_is_truncated(self) -> None:
        result = await terminal_tool.execute(
            command='python -c "print(\'x \' * 50000)"',
            timeout=5,
        )
        assert "... (truncated)" in result

    @pytest.mark.asyncio
    async def test_cat_file_within_workspace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        sample = tmp_path / "sample.txt"
        sample.write_text("hello terminal", encoding="utf-8")

        result = await terminal_tool.execute(command="cat sample.txt", timeout=5)
        assert "hello terminal" in result

    @pytest.mark.asyncio
    async def test_bare_env_var_is_blocked(self) -> None:
        result = await terminal_tool.execute(command="cat $HOME/.bashrc", timeout=5)
        assert result.startswith("Error:")
        assert "variable expansion" in result

    @pytest.mark.asyncio
    async def test_dollar_in_quotes_is_blocked(self) -> None:
        result = await terminal_tool.execute(command='ls "$PWD"', timeout=5)
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_python_c_os_system_is_blocked(self) -> None:
        result = await terminal_tool.execute(
            command='python -c "import os; os.system(\'id\')"', timeout=5,
        )
        assert result.startswith("Error:")
        assert "dangerous Python" in result

    @pytest.mark.asyncio
    async def test_python_c_subprocess_is_blocked(self) -> None:
        result = await terminal_tool.execute(
            command='python -c "import subprocess"', timeout=5,
        )
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_python_c_safe_code_allowed(self) -> None:
        result = await terminal_tool.execute(
            command='python -c "print(1+1)"', timeout=5,
        )
        assert "2" in result

    @pytest.mark.asyncio
    async def test_find_delete_is_blocked(self) -> None:
        result = await terminal_tool.execute(command="find . -delete", timeout=5)
        assert result.startswith("Error:")
        assert "dangerous argument" in result

    @pytest.mark.asyncio
    async def test_find_exec_is_blocked(self) -> None:
        result = await terminal_tool.execute(command="find . -exec cat {} +", timeout=5)
        assert result.startswith("Error:")
        assert "dangerous argument" in result

    @pytest.mark.asyncio
    async def test_stderr_to_dev_null_allowed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = await terminal_tool.execute(command="ls 2>/dev/null", timeout=5)
        assert not result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_fd_redirect_2_to_1_allowed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = await terminal_tool.execute(command="echo hello 2>&1", timeout=5)
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_unmatched_quote_error_message(self) -> None:
        result = await terminal_tool.execute(command='echo "hello', timeout=5)
        assert result.startswith("Error:")
        assert "syntax error" in result

    @pytest.mark.asyncio
    async def test_null_byte_is_blocked(self) -> None:
        result = await terminal_tool.execute(command="echo \x00", timeout=5)
        assert result.startswith("Error:")
        assert "null" in result
