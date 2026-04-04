"""
memory-v3 -- Structure Tags
Structure type definitions, auto-detection, and LLM hint generation.
"""

from __future__ import annotations

import re

STRUCTURE_TYPES = {
    "timeline": {
        "description": "Chronologically ordered events or history",
        "sort_key": "created_at",
        "hint": "These are timeline entries, presented in chronological order.",
    },
    "ledger": {
        "description": "Key-value pairs like config settings or environment variables",
        "sort_key": None,
        "hint": "These are ledger entries (key-value records). Each is a standalone fact.",
    },
    "taxonomy": {
        "description": "Hierarchical classification like architecture or org charts",
        "sort_key": None,
        "hint": "These are taxonomy entries organized hierarchically.",
    },
    "narrative": {
        "description": "Story-form content like conversation summaries or incident reports",
        "sort_key": "created_at",
        "hint": "These are narrative entries. They tell a story in sequence.",
    },
}

# Patterns that signal ledger-style content (key: value, key = value)
_LEDGER_PATTERNS = [
    re.compile(r"^[\w\s]+:\s+.+", re.MULTILINE),
    re.compile(r"^[\w\s]+=\s*.+", re.MULTILINE),
    re.compile(r"^\w+\s*->\s*.+", re.MULTILINE),
]

# Patterns that signal timeline content (dates, sequences)
_TIMELINE_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}"),
    re.compile(r"(first|then|next|after that|finally|subsequently)", re.IGNORECASE),
    re.compile(r"(on|at|by|since|until)\s+\w+\s+\d", re.IGNORECASE),
]

# Patterns that signal taxonomy/hierarchy content
_TAXONOMY_PATTERNS = [
    re.compile(r"^\s*[-*]\s+", re.MULTILINE),
    re.compile(r"(parent|child|subtype|category|class|hierarchy|inherits)", re.IGNORECASE),
    re.compile(r"^\s{2,}[-*]\s+", re.MULTILINE),  # indented bullets = hierarchy
]


def detect_structure_type(content: str) -> str:
    """
    Auto-detect structure type from content patterns.

    Checks for key-value patterns (ledger), date sequences (timeline),
    and hierarchy markers (taxonomy). Defaults to "narrative".
    """
    if not content:
        return "narrative"

    scores = {"ledger": 0, "timeline": 0, "taxonomy": 0, "narrative": 0}

    # Check ledger patterns
    for pattern in _LEDGER_PATTERNS:
        matches = pattern.findall(content)
        if len(matches) >= 2:
            scores["ledger"] += len(matches)

    # Check timeline patterns
    for pattern in _TIMELINE_PATTERNS:
        matches = pattern.findall(content)
        if matches:
            scores["timeline"] += len(matches)

    # Check taxonomy patterns
    for pattern in _TAXONOMY_PATTERNS:
        matches = pattern.findall(content)
        if matches:
            scores["taxonomy"] += len(matches)

    # Narrative gets a base score -- it's the default
    scores["narrative"] = 1

    best = max(scores, key=scores.get)
    return best


def get_structure_hint(structure_type: str, count: int) -> str:
    """
    Generate a hint string for LLM consumption.

    Combines the structure type's base hint with the result count.
    Falls back to a generic hint for unknown types.
    """
    info = STRUCTURE_TYPES.get(structure_type)
    if info is None:
        return f"Returning {count} memory entries."

    base_hint = info["hint"]
    return f"{base_hint} ({count} entries returned.)"
