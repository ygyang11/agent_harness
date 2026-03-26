"""Skill tool — loads domain knowledge from SKILL.md files on demand."""
from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Any

from agent_harness.core.config import HarnessConfig
from agent_harness.skills.loader import Skill, SkillLoader
from agent_harness.tool.base import BaseTool, ToolSchema

_DESCRIPTION_TEMPLATE = """\
Load a skill to get specialized instructions for the current task.

<available_skills>
{catalog}
</available_skills>

When to use:
- A user request clearly matches a skill description listed above
- You need domain-specific workflow guidance not in your training data

Rules:
- Use skill names exactly as listed above
- After loading, follow the skill instructions to complete the task
- Do not call this tool for skills that are already loaded in this conversation"""


def _format_resources(skill: Skill) -> str:
    resources = skill.list_resources()
    if not resources:
        return ""
    lines = ["\nAvailable resources:"]
    for folder, files in resources.items():
        names = ", ".join(f.name for f in files[:5])
        if len(files) > 5:
            names += f" ... ({len(files)} total)"
        lines.append(f"  - {folder}/: {names}")
    return "\n".join(lines) + "\n"


class SkillTool(BaseTool):
    """Built-in skill tool with config-driven skill directory loading."""

    _STATE_CHECK_INTERVAL: float = 30.0

    def __init__(self) -> None:
        super().__init__(
            name="skill_tool",
            description="Load a skill to get specialized instructions for the current task.",
        )
        self._loader: SkillLoader | None = None
        self._loader_dirs_key: tuple[str, ...] | None = None
        self._loader_state_key: tuple[str, ...] | None = None
        self._state_key_checked_at: float = 0.0

    def _ensure_loader(self, *, force: bool = False) -> SkillLoader:
        dirs: list[str | Path] = list(HarnessConfig.get().skill.dirs)
        dirs_key = tuple(str(d) for d in dirs)

        now = time.monotonic()
        ttl_valid = (now - self._state_key_checked_at) < self._STATE_CHECK_INTERVAL

        if (
            self._loader is not None
            and dirs_key == self._loader_dirs_key
            and ttl_valid
            and not force
        ):
            return self._loader

        state_key = SkillLoader.build_state_key(dirs)
        self._state_key_checked_at = now

        if (
            self._loader is None
            or dirs_key != self._loader_dirs_key
            or state_key != self._loader_state_key
        ):
            self._loader = SkillLoader(dirs)
            self._loader_dirs_key = dirs_key
            self._loader_state_key = state_key

        return self._loader

    def get_schema(self) -> ToolSchema:
        loader = self._ensure_loader()
        description = _DESCRIPTION_TEMPLATE.format(catalog=loader.get_catalog())
        return ToolSchema(
            name=self.name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to load.",
                    },
                    "args": {
                        "type": "string",
                        "description": "Optional arguments, replaces $ARGUMENTS in skill body.",
                        "default": "",
                    },
                },
                "required": ["skill_name"],
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        skill_name = str(kwargs.get("skill_name", "")).strip()
        args = str(kwargs.get("args", ""))
        if not skill_name:
            return "Error: skill_name is required."

        loader = self._ensure_loader()
        skill = loader.get_skill(skill_name)

        if skill is None:
            self._ensure_loader(force=True)
            return f"Skill '{skill_name}' not found. It may have been removed."

        content = skill.body
        if "$ARGUMENTS" in content:
            safe_args = html.escape(args, quote=False)
            content = content.replace("$ARGUMENTS", safe_args)

        resources_hint = _format_resources(skill)
        return (
            f'<skill-loaded name="{html.escape(skill.name)}">\n'
            f"{content}\n"
            f"{resources_hint}"
            f"</skill-loaded>"
        )


skill_tool = SkillTool()
