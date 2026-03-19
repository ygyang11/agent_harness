"""Tests for builtin notes tools."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_harness.tool.builtin.take_notes import list_notes, read_notes, take_notes
from agent_harness.utils.token_counter import count_tokens


def _notes_dir() -> Path:
    return Path(".notes")


@pytest.fixture(autouse=True)
def clean_notes_dir() -> None:
    root = _notes_dir()
    if root.exists():
        for file in root.glob("*.jsonl"):
            file.unlink()
    else:
        root.mkdir(parents=True, exist_ok=True)


class TestTakeNotes:
    @pytest.mark.asyncio
    async def test_take_notes_writes_note_file(self) -> None:
        result = await take_notes.execute(
            topic="Quantum Hardware",
            content="Error correction roadmap and milestones.",
            title="QEC Snapshot",
            source="web_search",
            tags="hardware,qec",
        )

        assert result.startswith("Note saved:")
        files = list(_notes_dir().glob("*.jsonl"))
        assert len(files) == 1

        rows = files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 1
        payload = json.loads(rows[0])
        assert payload["topic"] == "quantum_hardware"
        assert payload["title"] == "QEC Snapshot"
        assert payload["source"] == "web_search"
        assert payload["tags"] == ["hardware", "qec"]
        assert payload["created_at"] == payload["updated_at"]

    @pytest.mark.asyncio
    async def test_take_notes_rejects_empty_fields(self) -> None:
        empty_topic = await take_notes.execute(topic="", content="abc")
        empty_content = await take_notes.execute(topic="abc", content="")
        assert empty_topic == "Error: topic cannot be empty."
        assert empty_content == "Error: content cannot be empty."

    @pytest.mark.asyncio
    async def test_take_notes_truncates_long_content(self) -> None:
        long_text = "signal " * 20000
        await take_notes.execute(topic="long", content=long_text)

        files = list(_notes_dir().glob("*.jsonl"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text(encoding="utf-8").strip().splitlines()[0])
        assert count_tokens(payload["content"]) <= 10000

    @pytest.mark.asyncio
    async def test_take_notes_appends_for_same_topic(self) -> None:
        await take_notes.execute(topic="same", content="first")
        await take_notes.execute(topic="same", content="second")

        files = list(_notes_dir().glob("same_*.jsonl"))
        assert len(files) == 1
        rows = files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 2


class TestListNotes:
    @pytest.mark.asyncio
    async def test_list_notes_returns_no_data_when_empty(self) -> None:
        result = await list_notes.execute(limit=20)
        assert result == "No notes found."

    @pytest.mark.asyncio
    async def test_list_notes_returns_summary(self) -> None:
        await take_notes.execute(topic="chips", content="A")
        await take_notes.execute(topic="chips", content="B")

        result = await list_notes.execute(limit=20)
        assert "topic=chips" in result
        assert "count=2" in result
        assert ".notes" in result


class TestReadNotes:
    @pytest.mark.asyncio
    async def test_read_notes_returns_multiple_records(self) -> None:
        await take_notes.execute(topic="chips", content="first")
        await take_notes.execute(topic="chips", content="second")

        result = await read_notes.execute(topic="chips", limit=5)
        assert "topic=chips" in result
        assert "first" in result
        assert "second" in result

    @pytest.mark.asyncio
    async def test_read_notes_limit_applies(self) -> None:
        await take_notes.execute(topic="chips", content="one")
        await take_notes.execute(topic="chips", content="two")
        await take_notes.execute(topic="chips", content="three")

        result = await read_notes.execute(topic="chips", limit=2)
        assert result.count("[note_") == 2

    @pytest.mark.asyncio
    async def test_read_notes_returns_no_match(self) -> None:
        await take_notes.execute(topic="chips", content="one")
        result = await read_notes.execute(topic="biology", limit=5)
        assert result == "No matching notes."
