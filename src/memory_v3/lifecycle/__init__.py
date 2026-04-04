"""
memory-v3 -- Lifecycle Subpackage (Phases 5 + 6)

Full memory lifecycle: sensory gating, extraction, action decisions,
hallucination checking, consolidation, and compaction.

Public API
----------
ingest(conn, text, source, author, config)
    Full ingest pipeline: sensory filter -> extraction -> action decision -> write.

consolidate(conn, cycle_type, config)
    Run sleep consolidation cycle.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from ..config import Config, get_config


def ingest(
    conn: sqlite3.Connection,
    text: str,
    source: Optional[str] = None,
    author: str = "claude-code",
    config: Optional[Config] = None,
) -> dict:
    """Full ingest pipeline: sensory filter -> extraction -> action decision -> write.

    Returns a summary dict with counts for accepted, added, updated, deleted, skipped.
    """
    cfg = config or get_config()

    from .extraction import process_conversation

    result = process_conversation(
        conn, text, source=source, author=author, config=cfg,
    )
    return result


def consolidate(
    conn: sqlite3.Connection,
    cycle_type: str = "full",
    config: Optional[Config] = None,
) -> dict:
    """Run sleep consolidation cycle.

    cycle_type: 'replay', 'reorganize', 'compress', 'prune', or 'full'.
    Returns dict of per-phase stats.
    """
    cfg = config or get_config()

    from .consolidation import run_consolidation

    return run_consolidation(conn, cycle_type=cycle_type, config=cfg)
