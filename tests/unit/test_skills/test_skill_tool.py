from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness.skills.loader import SkillLoader
from agent_harness.tool.builtin.skill_tool import create_load_skill


class TestCreateLoadSkill:
    def test_description_contains_catalog(self, skills_dir: Path) -> None:
        loader = SkillLoader([skills_dir])
        tool = create_load_skill(loader)
        assert "<available_skills>" in tool.description
        assert "pdf" in tool.description
        assert "frontend" in tool.description

    def test_schema_parameters(self, skills_dir: Path) -> None:
        loader = SkillLoader([skills_dir])
        tool = create_load_skill(loader)
        schema = tool.get_schema()
        props = schema.parameters["properties"]
        assert "skill_name" in props
        assert "args" in props
        assert "skill_name" in schema.parameters["required"]


class TestLoadSkillExecution:
    @pytest.fixture
    def tool(self, skills_dir: Path):  # noqa: ANN201
        loader = SkillLoader([skills_dir])
        return create_load_skill(loader)

    async def test_execute_returns_skill_content(self, tool) -> None:  # noqa: ANN001
        result = await tool.execute(skill_name="pdf")
        assert '<skill-loaded name="pdf">' in result
        assert "Do PDF things." in result
        assert "</skill-loaded>" in result

    async def test_execute_not_found(self, tool) -> None:  # noqa: ANN001
        result = await tool.execute(skill_name="nonexistent")
        assert "not found" in result.lower()
        assert "pdf" in result

    async def test_execute_empty_name(self, tool) -> None:  # noqa: ANN001
        result = await tool.execute(skill_name="")
        assert "Error" in result

    async def test_arguments_substitution(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        sd = root / "tpl"
        sd.mkdir(parents=True)
        (sd / "SKILL.md").write_text(
            "---\nname: tpl\ndescription: Template skill.\n---\n\nHello $ARGUMENTS!\n"
        )
        loader = SkillLoader([root])
        tool = create_load_skill(loader)
        result = await tool.execute(skill_name="tpl", args="world")
        assert "Hello world!" in result

    async def test_resource_hints(self, tool) -> None:  # noqa: ANN001
        result = await tool.execute(skill_name="pdf")
        assert "scripts/" in result
        assert "extract.py" in result


class TestGetBuiltinToolsIntegration:
    def test_includes_load_skill(self, skills_dir: Path) -> None:
        from agent_harness.tool.builtin import get_builtin_tools

        tools = get_builtin_tools(skill_dirs=[skills_dir])
        names = [t.name for t in tools]
        assert "load_skill" in names

    def test_no_load_skill_when_empty(self, empty_skills_dir: Path) -> None:
        from agent_harness.tool.builtin import get_builtin_tools

        tools = get_builtin_tools(skill_dirs=[empty_skills_dir])
        names = [t.name for t in tools]
        assert "load_skill" not in names

    def test_no_load_skill_when_dir_missing(self, tmp_path: Path) -> None:
        from agent_harness.tool.builtin import get_builtin_tools

        tools = get_builtin_tools(skill_dirs=[tmp_path / "nope" / "skills"])
        names = [t.name for t in tools]
        assert "load_skill" not in names
