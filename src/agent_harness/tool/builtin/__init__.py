"""Builtin tools shipped with agent_harness."""
from __future__ import annotations

from pathlib import Path

from agent_harness.tool.base import BaseTool
from agent_harness.tool.builtin.paper_fetch import paper_fetch
from agent_harness.tool.builtin.paper_search import paper_search
from agent_harness.tool.builtin.pdf_parser import pdf_parser
from agent_harness.tool.builtin.take_notes import list_notes, read_notes, take_notes
from agent_harness.tool.builtin.terminal_tool import terminal_tool
from agent_harness.tool.builtin.web_fetch import web_fetch
from agent_harness.tool.builtin.web_search import web_search

BUILTIN_TOOLS: list[BaseTool] = [
    terminal_tool,
    web_fetch,
    web_search,
    pdf_parser,
    paper_search,
    paper_fetch,
    take_notes,
    list_notes,
    read_notes,
]


def get_builtin_tools(
    skill_dirs: list[str | Path] | None = None,
) -> list[BaseTool]:
    """Get all builtin tools, including load_skill if skills are available."""
    from agent_harness.skills.loader import SkillLoader
    from agent_harness.tool.builtin.skill_tool import create_load_skill

    tools = list(BUILTIN_TOOLS)
    dirs: list[str | Path]
    if skill_dirs is None:
        from agent_harness.core.config import HarnessConfig

        dirs = list(HarnessConfig.get().skill.dirs)
    else:
        dirs = list(skill_dirs)
    loader = SkillLoader(dirs)
    if loader.list_names():
        tools.append(create_load_skill(loader))
    return tools


__all__ = ["BUILTIN_TOOLS", "get_builtin_tools"]
