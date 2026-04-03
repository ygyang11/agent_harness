"""Tests for edit_file tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_app.tools.filesystem.edit_file import edit_file


class TestEditFile:
    @pytest.mark.asyncio
    async def test_single_replacement(self, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("def foo():\n    return 1\n")
        result = await edit_file.execute(
            file_path=str(f),
            old_string="return 1",
            new_string="return 42",
        )
        assert "1 replacement" in result
        assert f.read_text() == "def foo():\n    return 42\n"

    @pytest.mark.asyncio
    async def test_replace_all(self, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("x = 1\ny = 1\nz = 1\n")
        result = await edit_file.execute(
            file_path=str(f),
            old_string="1",
            new_string="2",
            replace_all=True,
        )
        assert "3 replacements" in result
        assert f.read_text() == "x = 2\ny = 2\nz = 2\n"

    @pytest.mark.asyncio
    async def test_ambiguous_match_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("x = 1\ny = 1\n")
        result = await edit_file.execute(
            file_path=str(f),
            old_string="1",
            new_string="2",
        )
        assert "appears 2 times" in result
        assert f.read_text() == "x = 1\ny = 1\n"

    @pytest.mark.asyncio
    async def test_empty_old_string_rejected(self, tmp_path: Path) -> None:
        result = await edit_file.execute(
            file_path=str(tmp_path / "any.txt"),
            old_string="",
            new_string="something",
        )
        assert "cannot be empty" in result

    @pytest.mark.asyncio
    async def test_old_equals_new_rejected(self, tmp_path: Path) -> None:
        result = await edit_file.execute(
            file_path=str(tmp_path / "any.txt"),
            old_string="same",
            new_string="same",
        )
        assert "identical" in result

    @pytest.mark.asyncio
    async def test_not_found(self, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("hello\n")
        result = await edit_file.execute(
            file_path=str(f),
            old_string="missing",
            new_string="new",
        )
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_sensitive_path_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / ".env"
        f.write_text("SECRET=old\n")
        result = await edit_file.execute(
            file_path=str(f),
            old_string="old",
            new_string="new",
        )
        assert "Refusing" in result
        assert f.read_text() == "SECRET=old\n"

    @pytest.mark.asyncio
    async def test_preserve_crlf(self, tmp_path: Path) -> None:
        f = tmp_path / "win.txt"
        f.write_bytes(b"line1\r\nline2\r\nline3\r\n")
        result = await edit_file.execute(
            file_path=str(f),
            old_string="line2",
            new_string="replaced",
        )
        assert "1 replacement" in result
        raw = f.read_bytes()
        assert b"replaced\r\n" in raw
        assert b"\r\n" in raw

    @pytest.mark.asyncio
    async def test_preserve_bom(self, tmp_path: Path) -> None:
        f = tmp_path / "bom.txt"
        f.write_bytes(b"\xef\xbb\xbfhello world\n")
        result = await edit_file.execute(
            file_path=str(f),
            old_string="hello",
            new_string="goodbye",
        )
        assert "1 replacement" in result
        assert f.read_bytes().startswith(b"\xef\xbb\xbf")

    @pytest.mark.asyncio
    async def test_non_utf8_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "gbk.txt"
        f.write_bytes(b"\xc4\xe3\xba\xc3")
        result = await edit_file.execute(
            file_path=str(f),
            old_string="x",
            new_string="y",
        )
        assert "not valid UTF-8" in result

    @pytest.mark.asyncio
    async def test_file_not_found(self, tmp_path: Path) -> None:
        result = await edit_file.execute(
            file_path=str(tmp_path / "nonexistent.py"),
            old_string="x",
            new_string="y",
        )
        assert "does not exist" in result

    @pytest.mark.asyncio
    async def test_diff_output(self, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("def foo():\n    return 1\n")
        result = await edit_file.execute(
            file_path=str(f),
            old_string="return 1",
            new_string="return 42",
        )
        assert "-    return 1" in result
        assert "+    return 42" in result
