from __future__ import annotations

import logging
from pathlib import Path

from agent_harness.skills.loader import SkillLoader


class TestSkillLoaderScan:
    def test_scan_valid_skills(self, skills_dir: Path) -> None:
        loader = SkillLoader([skills_dir])
        names = loader.list_names()
        assert "pdf" in names
        assert "frontend" in names

    def test_scan_multiple_dirs_priority(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a" / "skills"
        dir_b = tmp_path / "b" / "skills"
        for d, desc in [(dir_a, "first"), (dir_b, "second")]:
            sd = d / "dup"
            sd.mkdir(parents=True)
            (sd / "SKILL.md").write_text(
                f"---\nname: dup\ndescription: {desc}\n---\n\nbody\n"
            )
        loader = SkillLoader([dir_a, dir_b])
        skill = loader.get_skill("dup")
        assert skill is not None
        assert skill.description == "first"

    def test_scan_skip_invalid_missing_name(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        sd = root / "bad"
        sd.mkdir(parents=True)
        (sd / "SKILL.md").write_text("---\ndescription: no name\n---\n\nbody\n")
        loader = SkillLoader([root])
        assert loader.list_names() == []

    def test_scan_skip_no_frontmatter(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        sd = root / "plain"
        sd.mkdir(parents=True)
        (sd / "SKILL.md").write_text("# Just markdown, no frontmatter\n")
        loader = SkillLoader([root])
        assert loader.list_names() == []

    def test_scan_nonexistent_dir(self, tmp_path: Path) -> None:
        loader = SkillLoader([tmp_path / "nope" / "skills"])
        assert loader.list_names() == []

    def test_scan_empty_dir(self, empty_skills_dir: Path) -> None:
        loader = SkillLoader([empty_skills_dir])
        assert loader.list_names() == []


class TestSkillLoaderGetSkill:
    def test_get_skill_loads_body(self, skills_dir: Path) -> None:
        loader = SkillLoader([skills_dir])
        skill = loader.get_skill("pdf")
        assert skill is not None
        assert "Do PDF things." in skill.body

    def test_get_skill_loads_metadata(self, skills_dir: Path) -> None:
        loader = SkillLoader([skills_dir])
        skill = loader.get_skill("pdf")
        assert skill is not None
        assert skill.metadata.get("version") == 1.0

    def test_get_skill_caches(self, skills_dir: Path) -> None:
        loader = SkillLoader([skills_dir])
        s1 = loader.get_skill("pdf")
        s2 = loader.get_skill("pdf")
        assert s1 is s2

    def test_get_skill_not_found(self, skills_dir: Path) -> None:
        loader = SkillLoader([skills_dir])
        assert loader.get_skill("nonexistent") is None


class TestSkillLoaderCatalog:
    def test_get_catalog(self, skills_dir: Path) -> None:
        loader = SkillLoader([skills_dir])
        catalog = loader.get_catalog()
        assert "pdf" in catalog
        assert "Process PDF files." in catalog
        assert "version: 1.0" in catalog

    def test_get_catalog_empty(self, empty_skills_dir: Path) -> None:
        loader = SkillLoader([empty_skills_dir])
        assert loader.get_catalog() == "(no skills available)"

    def test_reload(self, skills_dir: Path) -> None:
        loader = SkillLoader([skills_dir])
        assert "pdf" in loader.list_names()
        (skills_dir / "pdf" / "SKILL.md").unlink()
        loader.reload()
        assert "pdf" not in loader.list_names()


class TestSkillLoaderStateKey:
    def test_build_state_key_changes_on_add(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        before = SkillLoader.build_state_key([root])

        sd = root / "new_skill"
        sd.mkdir()
        (sd / "SKILL.md").write_text(
            "---\nname: new_skill\ndescription: demo\n---\n\nbody\n",
            encoding="utf-8",
        )

        after = SkillLoader.build_state_key([root])
        assert before != after

    def test_build_state_key_changes_on_modify(self, skills_dir: Path) -> None:
        before = SkillLoader.build_state_key([skills_dir])
        md = skills_dir / "pdf" / "SKILL.md"
        md.write_text(
            "---\nname: pdf\ndescription: Process PDF files.\n---\n\nupdated body\n",
            encoding="utf-8",
        )
        after = SkillLoader.build_state_key([skills_dir])
        assert before != after


class TestSkillListResources:
    def test_list_resources_with_scripts(self, skills_dir: Path) -> None:
        loader = SkillLoader([skills_dir])
        skill = loader.get_skill("pdf")
        assert skill is not None
        resources = skill.list_resources()
        assert "scripts" in resources
        assert any(f.name == "extract.py" for f in resources["scripts"])

    def test_list_resources_empty(self, skills_dir: Path) -> None:
        loader = SkillLoader([skills_dir])
        skill = loader.get_skill("frontend")
        assert skill is not None
        assert skill.list_resources() == {}


class TestSkillLoaderNameCollisionWarning:
    def test_duplicate_name_logs_warning(self, tmp_path: Path, caplog: logging.LogCaptureFixture) -> None:
        dir_a = tmp_path / "a" / "skills"
        dir_b = tmp_path / "b" / "skills"
        for d in (dir_a, dir_b):
            sd = d / "dup"
            sd.mkdir(parents=True)
            (sd / "SKILL.md").write_text(
                "---\nname: dup\ndescription: duplicate\n---\n\nbody\n"
            )
        with caplog.at_level(logging.WARNING, logger="agent_harness.skills.loader"):
            SkillLoader([dir_a, dir_b])
        assert any("shadows" in record.message for record in caplog.records)
