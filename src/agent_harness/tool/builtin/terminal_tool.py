"""Unified terminal tool for workspace-restricted command execution."""
from __future__ import annotations

import asyncio
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from agent_harness.tool.decorator import tool
from agent_harness.utils.token_counter import truncate_text_by_tokens


@dataclass(frozen=True)
class TerminalSafetyConfig:
    """Centralized security policy for the terminal tool.

    All validation constants are grouped here for maintainability.
    Frozen to prevent runtime mutation.
    """

    # -- Output limits --
    max_command_chars: int = 64_000
    max_output_tokens: int = 15_000

    # -- Shell operators --
    chain_ops: frozenset[str] = frozenset({"&&", "||", ";"})
    unsupported_ops: frozenset[str] = frozenset({"|", "&"})
    supported_redirect_ops: frozenset[str] = frozenset({">", ">>", "<"})
    unsupported_redirect_ops: frozenset[str] = frozenset({"<<", "<<<", "<>", ">|"})
    fd_redirect_ops: frozenset[str] = frozenset({">&", "<&"})
    safe_redirect_targets: frozenset[str] = frozenset({"/dev/null"})

    # -- Command allowlists / blocklists --
    allowed_commands: frozenset[str] = frozenset({
        "ls", "cat", "head", "tail", "find", "grep",
        "wc", "sort", "uniq", "echo", "printf", "pwd",
        "python", "python3",
    })
    blocked_executors: frozenset[str] = frozenset({
        "sh", "bash", "zsh", "source", ".",
    })
    dangerous_tokens: frozenset[str] = frozenset({
        "rm", "sudo", "chmod", "chown", "mkfs", "dd",
        "shutdown", "reboot",
    })
    no_path_args_commands: frozenset[str] = frozenset({
        "pwd", "echo", "printf",
    })
    python_commands: frozenset[str] = frozenset({"python", "python3"})

    # -- Per-command dangerous arguments --
    dangerous_command_args: dict[str, frozenset[str]] = field(default_factory=lambda: {
        "find": frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir"}),
        "sort": frozenset({"-o"}),
    })

    # -- Regex patterns --
    # Matches bare $VAR references (but not ${} or $())
    bare_var_re: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"\$(?![({])([A-Za-z_])")
    )
    # Dangerous Python stdlib calls that can escape the workspace
    dangerous_python_re: re.Pattern[str] = field(
        default_factory=lambda: re.compile(
            r"(?:"
            r"os\.(?:system|popen|exec|remove|unlink|rmdir|rename|chmod|chown)"
            r"|subprocess"
            r"|shutil\.(?:rmtree|move)"
            r"|open\s*\(.*['\"]/"
            r"|__import__"
            r"|importlib"
            r"|eval\s*\("
            r"|exec\s*\("
            r"|compile\s*\("
            r")",
            re.IGNORECASE,
        )
    )


_SAFETY = TerminalSafetyConfig()


def _workspace_root() -> Path:
    return Path.cwd().resolve()


def _resolve_workspace_path(
    path_value: str,
    *,
    workspace_root: Path,
    cwd_path: Path | None = None,
    must_exist: bool = False,
    must_be_directory: bool = False,
) -> Path:
    raw = Path(path_value)
    base = cwd_path if cwd_path is not None else workspace_root
    resolved = raw.resolve() if raw.is_absolute() else (base / raw).resolve()

    try:
        resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {path_value}") from exc

    if must_exist and not resolved.exists():
        raise ValueError(f"path does not exist: {path_value}")

    if must_be_directory and not resolved.is_dir():
        raise ValueError(f"path is not a directory: {path_value}")

    return resolved


def _split_command_chain(command: str) -> list[list[str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|><")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError as exc:
        raise ValueError(f"command syntax error: {exc}") from exc

    if not tokens:
        raise ValueError("empty command")

    segments: list[list[str]] = []
    current: list[str] = []

    for token in tokens:
        if token in _SAFETY.chain_ops:
            if not current:
                raise ValueError("invalid command chain")
            segments.append(current)
            current = []
            continue

        if token in _SAFETY.unsupported_ops:
            raise ValueError(f"unsupported shell operator: {token}")

        current.append(token)

    if not current:
        raise ValueError("invalid command chain")

    segments.append(current)
    return segments


def _validate_shell_safety(command: str) -> None:
    if "\x00" in command:
        raise ValueError("null bytes are not allowed in commands")

    lowered = command.lower()
    if "${" in command or "$(" in command or "`" in command:
        raise ValueError("dynamic shell expansion is not allowed")

    # Bare $VAR expands at bash runtime but is invisible to static shlex
    # parsing, creating a gap between validation and execution semantics.
    if _SAFETY.bare_var_re.search(command):
        raise ValueError("shell variable expansion is not allowed")

    if "eval " in lowered or lowered.startswith("eval"):
        raise ValueError("eval is not allowed")

    segments = _split_command_chain(command)

    for segment in segments:
        for token in segment:
            if token.lower() in _SAFETY.dangerous_tokens:
                raise ValueError(f"disallowed token detected: {token}")


def _validate_path_token(
    token: str,
    *,
    workspace_root: Path,
    cwd_path: Path,
    must_exist: bool = False,
) -> None:
    if token.startswith("~"):
        raise ValueError(f"path escapes workspace: {token}")

    should_treat_as_path = (
        token.startswith(".")
        or token.startswith("/")
        or "/" in token
        or any(char in token for char in ("*", "?", "["))
    )
    if not should_treat_as_path:
        return

    if any(char in token for char in ("*", "?", "[")):
        parent = Path(token).parent
        parent_token = "." if str(parent) in {"", "."} else str(parent)
        _resolve_workspace_path(
            parent_token,
            workspace_root=workspace_root,
            cwd_path=cwd_path,
        )
        return

    _resolve_workspace_path(
        token,
        workspace_root=workspace_root,
        cwd_path=cwd_path,
        must_exist=must_exist,
    )


def _validate_python_segment(
    segment: list[str],
    *,
    workspace_root: Path,
    cwd_path: Path,
) -> None:
    if "-c" in segment[1:]:
        c_index = segment.index("-c", 1)
        code_arg = segment[c_index + 1] if c_index + 1 < len(segment) else ""
        if _SAFETY.dangerous_python_re.search(code_arg):
            raise ValueError("dangerous Python operation detected")
        return

    for arg in segment[1:]:
        if arg.startswith("-"):
            continue
        _validate_path_token(
            arg,
            workspace_root=workspace_root,
            cwd_path=cwd_path,
            must_exist=True,
        )
        return


def _validate_redirections(
    segment: list[str],
    *,
    workspace_root: Path,
    cwd_path: Path,
) -> set[int]:
    skip_indexes: set[int] = set()
    index = 1
    while index < len(segment):
        token = segment[index]
        if token in _SAFETY.unsupported_redirect_ops:
            raise ValueError(f"unsupported redirection operator: {token}")

        if token in _SAFETY.fd_redirect_ops:
            target_index = index + 1
            if target_index >= len(segment):
                raise ValueError(f"missing fd target after {token}")
            target = segment[target_index]
            if not target.isdigit():
                raise ValueError(f"fd redirect target must be a number: {target}")
            skip_indexes.add(index)
            skip_indexes.add(target_index)
            index += 2
            continue

        if token in _SAFETY.supported_redirect_ops:
            target_index = index + 1
            if target_index >= len(segment):
                raise ValueError(f"missing redirect target after {token}")
            target = segment[target_index]

            # Mark preceding fd number (e.g. '2' in ['2', '>', '/dev/null'])
            if index > 1 and segment[index - 1].isdigit():
                skip_indexes.add(index - 1)

            if target not in _SAFETY.safe_redirect_targets:
                _validate_path_token(
                    target,
                    workspace_root=workspace_root,
                    cwd_path=cwd_path,
                    must_exist=(token == "<"),
                )
            skip_indexes.add(index)
            skip_indexes.add(target_index)
            index += 2
            continue

        index += 1

    return skip_indexes


def _validate_command(
    command: str,
    *,
    workspace_root: Path,
    cwd_path: Path,
) -> None:
    if len(command) > _SAFETY.max_command_chars:
        raise ValueError(f"command too large (>{_SAFETY.max_command_chars} chars)")

    _validate_shell_safety(command)
    segments = _split_command_chain(command)

    for segment in segments:
        base = segment[0].lower()
        skip_indexes = _validate_redirections(
            segment,
            workspace_root=workspace_root,
            cwd_path=cwd_path,
        )

        if base in _SAFETY.blocked_executors:
            raise ValueError(f"blocked command executor: {base}")
        if base not in _SAFETY.allowed_commands:
            raise ValueError(f"command not allowed: {base}")

        if base in _SAFETY.no_path_args_commands:
            continue

        if base in _SAFETY.python_commands:
            _validate_python_segment(
                segment,
                workspace_root=workspace_root,
                cwd_path=cwd_path,
            )
            continue

        dangerous_args = _SAFETY.dangerous_command_args.get(base, frozenset())
        for arg_index, arg in enumerate(segment[1:], start=1):
            if arg_index in skip_indexes:
                continue
            if arg.lower() in dangerous_args:
                raise ValueError(f"dangerous argument for {base}: {arg}")
            if arg.startswith("-"):
                continue
            _validate_path_token(
                arg,
                workspace_root=workspace_root,
                cwd_path=cwd_path,
            )


def _normalize_output(stdout_text: str, stderr_text: str) -> str:
    merged = stdout_text if not stderr_text else f"{stdout_text}\n{stderr_text}".strip()
    if not merged:
        return ""

    return truncate_text_by_tokens(
        merged,
        max_tokens=_SAFETY.max_output_tokens,
        suffix="\n... (truncated)",
    )


@tool
async def terminal_tool(command: str, timeout: int = 30, cwd: str = ".") -> str:
    """Execute a workspace-restricted shell command in a bash subprocess.

    Runs allowlisted commands (ls, cat, grep, python, etc.) in a non-login
    bash subprocess. All paths must resolve within the workspace root.
    Shell variable expansion and destructive commands are blocked.
    Supports chaining (&&, ||, ;) and redirection (>, >>, <).

    This is not an OS-level sandbox — security relies on static
    pre-execution validation of the command string.

    Args:
        command: Shell command line, e.g. ``ls -la``, ``grep -r TODO .``, ``python -c "print(1)"``. Chaining (&&, ||, ;) and I/O redirection are supported.
        timeout: Maximum execution time in seconds (positive integer, default 30).
        cwd: Working directory for execution, as a relative path within the workspace (e.g. ``"src/utils"``). Defaults to workspace root.

    Returns:
        Command stdout/stderr text, truncated to token budget. Non-zero
        exits prefixed with ``[exit code N]``, errors with ``Error:``.
    """
    if not command.strip():
        return "Error: command cannot be empty"

    if timeout <= 0:
        return "Error: timeout must be greater than 0"

    workspace_root = _workspace_root()

    try:
        cwd_path = _resolve_workspace_path(
            cwd,
            workspace_root=workspace_root,
            must_exist=True,
            must_be_directory=True,
        )
        _validate_command(
            command,
            workspace_root=workspace_root,
            cwd_path=cwd_path,
        )
    except ValueError as exc:
        return f"Error: {exc}"

    try:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=str(cwd_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"Error: execution timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return f"Error: failed to execute command: {exc}"

    stdout_text = stdout.decode(errors="replace")
    stderr_text = stderr.decode(errors="replace")
    output = _normalize_output(stdout_text, stderr_text)

    if proc.returncode != 0:
        return f"[exit code {proc.returncode}]\n{output}".rstrip()
    return output or "(no output)"
