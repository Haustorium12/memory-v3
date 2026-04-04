"""
memory-v3 -- Stage 1: Coarse Candidate Filtering
Fast, broad retrieval to build a candidate pool for fine scoring.

Pipeline:
  1. Route query to relevant graph layers
  2. Find top communities by centroid similarity
  3. Collect memory IDs from those communities
  4. Fast vector search (vec0)
  5. FTS5 keyword search
  6. Union all candidate IDs, cap at STAGE1_MAX_CANDIDATES
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)


def _serialize_f32(vec: list[float]) -> bytes:
    """Serialize a list of floats to raw bytes for sqlite-vec."""
    return np.array(vec, dtype=np.float32).tobytes()


def _escape_fts5_query(query: str) -> str:
    """
    Escape a raw query string for safe use in FTS5 MATCH.

    Wraps each word in double quotes to prevent FTS5 syntax errors
    from special characters (colons, hyphens, etc.).
    """
    words = query.split()
    if not words:
        return '""'
    escaped = " OR ".join(f'"{w}"' for w in words if w.strip())
    return escaped or '""'


def _get_community_candidate_ids(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    config: "Config",
) -> set[int]:
    """
    Find top communities by centroid similarity and return all memory IDs
    belonging to those communities.
    """
    candidate_ids: set[int] = set()

    try:
        from ..graphs.communities import get_community_for_query

        top_n = getattr(config, "stage1_community_top", 3)
        community_ids = get_community_for_query(query_embedding, conn, top_n=top_n)

        if not community_ids:
            return candidate_ids

        # Collect all memory IDs with matching cluster_id
        placeholders = ",".join("?" for _ in community_ids)
        rows = conn.execute(
            f"SELECT id FROM memories WHERE cluster_id IN ({placeholders}) AND archived = 0",
            community_ids,
        ).fetchall()

        for row in rows:
            candidate_ids.add(row[0] if isinstance(row, tuple) else row["id"])

    except Exception as exc:
        logger.debug("Community candidate lookup skipped: %s", exc)

    return candidate_ids


def _get_vector_candidate_ids(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    limit: int = 200,
) -> list[int]:
    """
    Fast vector similarity search via the vec0 virtual table.

    Returns memory IDs ordered by ascending distance (most similar first).
    """
    try:
        rows = conn.execute(
            "SELECT id, distance FROM memory_vec "
            "WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT ?",
            (_serialize_f32(query_embedding), limit),
        ).fetchall()
        return [row[0] if isinstance(row, tuple) else row["id"] for row in rows]
    except Exception as exc:
        logger.debug("Vector search failed: %s", exc)
        return []


def _get_fts_candidate_ids(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 200,
) -> list[int]:
    """
    FTS5 keyword search for candidate memory IDs.

    Returns rowids ordered by FTS5 rank (best match first).
    """
    escaped = _escape_fts5_query(query)
    try:
        rows = conn.execute(
            "SELECT rowid, rank FROM memory_fts "
            "WHERE memory_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (escaped, limit),
        ).fetchall()
        return [row[0] if isinstance(row, tuple) else row["rowid"] for row in rows]
    except Exception as exc:
        logger.debug("FTS5 search failed: %s", exc)
        return []


def get_candidates(
    conn: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    config: "Config",
) -> list[int]:
    """
    Stage 1: Build a broad candidate pool from multiple retrieval channels.

    Combines community-based, vector, and keyword candidates into a single
    deduplicated list, capped at config.stage1_max_candidates.

    Args:
        conn: SQLite connection with memory tables.
        query: The raw user query string.
        query_embedding: Pre-computed embedding vector for the query.
        config: The memory-v3 Config object.

    Returns:
        List of unique memory IDs (ints), up to stage1_max_candidates.
    """
    max_candidates = getattr(config, "stage1_max_candidates", 500)

    # --- Channel 1: Community-routed candidates ---
    community_ids = _get_community_candidate_ids(conn, query_embedding, config)

    # --- Channel 2: Vector similarity candidates ---
    vec_ids = _get_vector_candidate_ids(conn, query_embedding, limit=200)

    # --- Channel 3: FTS5 keyword candidates ---
    fts_ids = _get_fts_candidate_ids(conn, query, limit=200)

    # --- Union all channels, preserving rough ordering ---
    # Vector and FTS results are ordered by relevance; community IDs are unordered.
    # We use an ordered set approach: vec first, then FTS, then community.
    seen: set[int] = set()
    ordered: list[int] = []

    for mid in vec_ids:
        if mid not in seen:
            seen.add(mid)
            ordered.append(mid)

    for mid in fts_ids:
        if mid not in seen:
            seen.add(mid)
            ordered.append(mid)

    for mid in community_ids:
        if mid not in seen:
            seen.add(mid)
            ordered.append(mid)

    # Cap at max_candidates
    result = ordered[:max_candidates]

    logger.debug(
        "Stage 1 candidates: %d (vec=%d, fts=%d, community=%d, deduped=%d)",
        len(result), len(vec_ids), len(fts_ids), len(community_ids), len(ordered),
    )

    return result
