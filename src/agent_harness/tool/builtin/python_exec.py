"""Python code execution tool."""
from __future__ import annotations

import asyncio
import os
import tempfile

from agent_harness.tool.decorator import tool


@tool
async def python_exec(code: str, timeout: int = 30) -> str:
    """Execute a Python code snippet written to a temporary file in a subprocess.

    The code is written to a temporary file, executed with the current Python
    interpreter, and stdout/stderr are captured and returned.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        Combined stdout and stderr output, or an error/timeout message.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            fh.write(code)

        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Error: execution timed out after {timeout}s"
        except Exception as exc:  # noqa: BLE001
            return f"Error launching subprocess: {exc}"

        output_parts: list[str] = []
        if stdout:
            output_parts.append(stdout.decode(errors="replace"))
        if stderr:
            output_parts.append(stderr.decode(errors="replace"))

        result = "\n".join(output_parts).strip()
        if proc.returncode != 0:
            result = f"[exit code {proc.returncode}]\n{result}"
        return result if result else "(no output)"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
