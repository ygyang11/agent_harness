"""Structured note management tools."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from agent_harness.tool.decorator import tool
from agent_harness.utils.token_counter import truncate_text_by_tokens

_MAX_NOTE_TOKENS = 10000
_MAX_PREVIEW_TOKENS = 100
_NOTES_ROOT = Path(".notes")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "general"


def _resolve_topic_file(topic: str) -> Path:
    day = datetime.now().strftime("%Y%m%d")
    return _NOTES_ROOT / f"{_slugify(topic)}_{day}.jsonl"


@tool
def take_notes(topic: str, content: str, title: str = "", source: str = "", tags: str = "") -> str:
    """Create or update a structured note to record a topic.

    Args:
        topic: Topic name used for grouping and file naming.
        content: Main note content.
        title: Optional short title.
        source: Optional source label (e.g. web_search).
        tags: Optional comma-separated tags.

    Returns:
        A summary string with note id, file path, and preview, or an error message.
    """
    clean_topic = topic.strip()
    clean_content = content.strip()
    if not clean_topic:
        return "Error: topic cannot be empty."
    if not clean_content:
        return "Error: content cannot be empty."

    now = datetime.now().isoformat()
    target = _resolve_topic_file(clean_topic)
    note_content = truncate_text_by_tokens(
        clean_content,
        max_tokens=_MAX_NOTE_TOKENS,
        suffix="...",
    )
    note_tags = [item.strip() for item in tags.split(",") if item.strip()]

    payload: dict[str, str | list[str]] = {
        "id": f"note_{uuid.uuid4().hex[:12]}",
        "topic": _slugify(clean_topic),
        "title": title.strip(),
        "content": note_content,
        "source": source.strip(),
        "tags": note_tags,
        "created_at": now,
        "updated_at": now,
    }

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:
        return f"Error: cannot persist note - {exc}"

    total = 0
    with target.open("r", encoding="utf-8") as f:
        for _ in f:
            total += 1

    preview = truncate_text_by_tokens(note_content, max_tokens=_MAX_PREVIEW_TOKENS, suffix="...")
    return (
        f"Note saved: id={payload['id']}, topic={payload['topic']}, "
        f"file={target}, total_in_file={total}, preview={preview}"
    )


@tool
def list_notes(limit: int = 20) -> str:
    """List available notes with file-level metadata.

    Args:
        limit: Maximum number of note files to summarize.

    Returns:
        A newline-separated summary list or a no-data message.
    """
    if not _NOTES_ROOT.exists():
        return "No notes found."

    files = sorted(_NOTES_ROOT.glob("*.jsonl"))
    if not files:
        return "No notes found."

    lines: list[str] = []
    for file in files[:limit]:
        count = 0
        latest = ""
        topic = file.stem
        with file.open("r", encoding="utf-8") as f:
            for raw in f:
                if not raw.strip():
                    continue
                count += 1
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                topic = str(item.get("topic", topic))
                latest = str(item.get("updated_at", item.get("created_at", latest)))
        lines.append(f"- topic={topic}, file={file}, count={count}, latest={latest}")
    return "\n".join(lines)


@tool
def read_notes(topic: str = "", limit: int = 5) -> str:
    """Read specific notes filtered by the given topic.

    Args:
        topic: Topic filter (substring match). Empty means all topics.
        limit: Maximum number of matched notes to return.

    Returns:
        Aggregated note contents or a no-match message.
    """
    if not _NOTES_ROOT.exists():
        return "No notes found."

    topic_filter = _slugify(topic) if topic.strip() else ""
    matched: list[str] = []

    for file in sorted(_NOTES_ROOT.glob("*.jsonl")):
        with file.open("r", encoding="utf-8") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                item_topic = str(item.get("topic", "")).lower()
                if topic_filter and topic_filter not in item_topic:
                    continue
                content = str(item.get("content", ""))
                matched.append(
                    f"[{item.get('id', '')}] topic={item.get('topic', '')} "
                    f"updated_at={item.get('updated_at', '')}\n{content}"
                )
                if len(matched) >= limit:
                    break
        if len(matched) >= limit:
            break

    if not matched:
        return "No matching notes."
    return "\n\n---\n\n".join(matched)
