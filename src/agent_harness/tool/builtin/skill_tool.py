"""Skill tool — loads domain knowledge from SKILL.md files on demand."""
from __future__ import annotations

from agent_harness.skills.loader import Skill, SkillLoader
from agent_harness.tool.base import ToolSchema
from agent_harness.tool.decorator import FunctionTool

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


def create_load_skill(loader: SkillLoader) -> FunctionTool:
    """Create a load_skill FunctionTool with dynamically embedded skill catalog."""
    catalog = loader.get_catalog()
    description = _DESCRIPTION_TEMPLATE.format(catalog=catalog)

    def load_skill(skill_name: str, args: str = "") -> str:
        """Load a skill by name.

        Args:
            skill_name: Name of the skill to load.
            args: Optional arguments, replaces $ARGUMENTS in the skill body.
        """
        if not skill_name:
            return "Error: skill_name is required."

        skill = loader.get_skill(skill_name)
        if skill is None:
            available = ", ".join(loader.list_names())
            return f"Skill '{skill_name}' not found. Available: {available}"

        content = skill.body
        if "$ARGUMENTS" in content:
            content = content.replace("$ARGUMENTS", args)

        resources_hint = _format_resources(skill)
        return (
            f'<skill-loaded name="{skill.name}">\n'
            f"{content}\n"
            f"{resources_hint}"
            f"</skill-loaded>"
        )

    schema = ToolSchema(
        name="load_skill",
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

    return FunctionTool(
        fn=load_skill,
        name="load_skill",
        description=description,
        schema=schema,
    )
