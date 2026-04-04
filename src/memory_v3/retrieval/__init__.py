"""
memory-v3 -- Retrieval Subpackage
3-stage search pipeline: coarse filtering, fine scoring, organization.

Public API
----------
search(conn, query, query_embedding, limit=10, content_type=None, config=None)
    Full 3-stage retrieval pipeline with graph-aware organization.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING, Optional

from . import stage1_coarse, stage2_fine, stage3_organize

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "search",
    "stage1_coarse",
    "stage2_fine",
    "stage3_organize",
]


def search(
    conn: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    limit: int = 10,
    content_type: Optional[str] = None,
    config: Optional["Config"] = None,
) -> list[dict]:
    """
    Full 3-stage retrieval pipeline.

    Stage 1 (Coarse): Fast candidate filtering via communities + vec + FTS5.
    Stage 2 (Fine):   RRF fusion + ACT-R + fan effect + confidence scoring.
    Stage 3 (Organize): Graph-aware ordering, neighbor annotation, structure hints.

    Falls back to v2-style direct hybrid search when three-stage mode is disabled.

    Args:
        conn: SQLite connection with full memory-v3 schema.
        query: The raw user query string.
        query_embedding: Pre-computed embedding vector for the query.
        limit: Maximum number of results to return.
        content_type: Optional filter by content_type (fact, episode, etc.).
        config: Optional Config override. Uses get_config() if None.

    Returns:
        List of scored, organized memory dicts with graph annotations.
    """
    from ..config import get_config

    cfg = config or get_config()

    if getattr(cfg, "enable_three_stage", True):
        # Stage 1: Coarse filter
        candidates = stage1_coarse.get_candidates(conn, query, query_embedding, cfg)

        if not candidates:
            logger.debug("Stage 1 returned no candidates; falling back to direct search")
            return _fallback_search(conn, query, query_embedding, limit, content_type)

        # Stage 2: Fine scoring on candidates
        scored = stage2_fine.score_candidates(conn, candidates, query, query_embedding, cfg)

        # Apply content_type filter if requested
        if content_type:
            scored = [r for r in scored if r.get("content_type") == content_type]

        # Stage 3: Organize results
        organized = stage3_organize.organize(conn, scored[:limit], query, cfg)
        return organized

    else:
        # Fallback: v2-style direct search
        return _fallback_search(conn, query, query_embedding, limit, content_type)


def _fallback_search(
    conn: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    limit: int,
    content_type: Optional[str],
) -> list[dict]:
    """v2-compatible direct hybrid search with scoring."""
    from .. import db
    from ..scoring import rank_results

    results = db.hybrid_search(
        conn, query, query_embedding, limit=limit, content_type=content_type,
    )
    return rank_results(conn, results, query)
