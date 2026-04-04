"""Terminal tool for shell command execution.

Security model — three layers:
- Layer 1 (Permission): ApprovalPolicy decides if execution is allowed
- Layer 2 (Sandbox): Not implemented yet (Phase 3)
- Layer 3 (Tool): Pure execution — no command filtering

The tool layer handles:
- Timeout enforcement and subprocess cleanup
- Output normalization and truncation

Commands run from the workspace root with full system access.
Security is enforced by ApprovalPolicy, not by the tool layer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent_harness.tool.decorator import tool
from agent_harness.utils.token_counter import truncate_text_by_tokens

_MAX_OUTPUT_TOKENS: int = 10_000
_DEFAULT_TIMEOUT: int = 120  # seconds
_MAX_TIMEOUT: int = 600  # seconds (10 minutes)

TERMINAL_TOOL_DESCRIPTION = (
    "Execute a shell command in a bash subprocess and return its output.\n\n"
    "The working directory defaults to the workspace root. "
    "Shell state (variables, aliases) does not persist between calls — "
    "each invocation starts a fresh subprocess.\n\n"
    "Use this tool for operations without a dedicated equivalent: "
    "git, pytest, pip, npm, make, docker, curl, python/node/bash scripts, etc.\n\n"
    "Examples:\n"
    "  Good:\n"
    "    terminal_tool(command='pytest tests/ -v')\n"
    "    terminal_tool(command='python scripts/migrate.py')\n"
    '    terminal_tool(command=\'git add -A && git commit -m "fix bug"\')\n'
    "    terminal_tool(command='cd src && python -m mymodule', timeout=300)\n"
    "  Bad:\n"
    "    terminal_tool(command='cat file.txt')       # use dedicated file reading tool\n"
    "    terminal_tool(command='grep -r pattern .')   # use dedicated search tool\n"
    "    terminal_tool(command='find . -name *.py')   # use dedicated file search tool\n\n"
    "Guidelines:\n"
    "- Always quote file paths containing spaces with double quotes\n"
    "- If a command creates files or directories, verify the parent exists first\n"
    "- Chain dependent commands with && (e.g. 'cd src && pytest')\n"
    "- Use ; only when you don't care if earlier commands fail\n"
    "- Set appropriate timeout for long-running commands (default 120s, max 600s)"
)


def _workspace_root() -> Path:
    """Return resolved cwd as workspace root."""
    return Path.cwd().resolve()


async def _execute_command(
    command: str,
    cwd: Path,
    timeout: int,
) -> tuple[int | None, str]:
    """Execute a shell command and return (exit_code, normalized_output).

    exit_code is None on timeout or execution failure.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return None, f"Error: execution timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return None, f"Error: failed to execute command: {exc}"

    stdout_text = stdout.decode(errors="replace")
    stderr_text = stderr.decode(errors="replace")
    merged = stdout_text if not stderr_text else f"{stdout_text}\n{stderr_text}".strip()

    if not merged:
        return proc.returncode, ""

    output = truncate_text_by_tokens(
        merged,
        max_tokens=_MAX_OUTPUT_TOKENS,
        suffix="\n... (truncated)",
    )
    return proc.returncode, output


@tool(description=TERMINAL_TOOL_DESCRIPTION, executor_timeout=_MAX_TIMEOUT + 10)
async def terminal_tool(command: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Execute a shell command in a bash subprocess.

    Args:
        command: Shell command to execute. Supports full bash syntax
            including pipes, redirects, chaining, and variable expansion.
        timeout: Maximum execution time in seconds (default 120, max 600).
    """
    if not command.strip():
        return "Error: command cannot be empty"

    if timeout <= 0:
        return "Error: timeout must be greater than 0"
    timeout = min(timeout, _MAX_TIMEOUT)

    exit_code, output = await _execute_command(command, _workspace_root(), timeout)

    if exit_code is None:
        return output

    if exit_code != 0:
        return f"[exit code {exit_code}]\n{output}".rstrip()

    return output or "(no output)"
