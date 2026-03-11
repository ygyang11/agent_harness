"""JSON utility functions for agent_harness.

Handles common issues when parsing JSON from LLM outputs:
- Markdown code block wrapping
- Trailing commas
- Partial/malformed JSON extraction
"""
from __future__ import annotations

import json
import re
from typing import Any


def parse_json_lenient(text: str) -> Any:
    """Parse JSON from text, handling common LLM output issues.

    Attempts:
    1. Direct JSON parse
    2. Extract from markdown code blocks
    3. Find first JSON object/array in text
    """
    # Attempt 1: direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract from markdown code block
    extracted = extract_code_block(text)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    # Attempt 3: find first JSON object or array
    extracted = extract_json_object(text)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from text: {text[:200]}...")


def extract_code_block(text: str) -> str | None:
    """Extract content from a markdown code block."""
    pattern = r"```(?:json)?\s*\n?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_json_object(text: str) -> str | None:
    """Extract the first JSON object or array from text using brace/bracket matching."""
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """JSON serialize with defaults for common types."""
    return json.dumps(obj, default=str, ensure_ascii=False, **kwargs)
