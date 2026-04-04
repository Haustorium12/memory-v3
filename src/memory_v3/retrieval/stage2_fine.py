"""
memory-v3 -- Stage 2: Fine-Grained Scoring
Applies RRF fusion, ACT-R activation, fan effect, confidence, and surprise
scoring to the candidate set produced by Stage 1.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)


def _serialize_f32(vec: list[float]) -> bytes:
    """Serialize a list of floats to raw bytes for sqlite-vec."""
    return np.array(vec, dtype=np.float32).tobytes()


def _escape_fts5_query(query: str) -> str:
    """Escape a raw query string for safe FTS5 MATCH."""
    words = query.split()
    if not words:
        return '""'
    return " OR ".join(f'"{w}"' for w in words if w.strip()) or '""'


def _fetch_memories(
    conn: sqlite3.Connection,
    candidate_ids: list[int],
) -> dict[int, dict]:
    """
    Fetch full memory records for a batch of candidate IDs.

    Uses batched IN queries to avoid hitting SQLite variable limits.
    Returns a dict mapping memory_id -> memory dict.
    """
    memories: dict[int, dict] = {}
    batch_size = 500

    for i in range(0, len(candidate_ids), batch_size):
        batch = candidate_ids[i : i + batch_size]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders}) AND archived = 0",
            batch,
        ).fetchall()
        for row in rows:
            mem = dict(row)
            # Parse JSON fields
            if isinstance(mem.get("tags"), str):
                try:
                    mem["tags"] = json.loads(mem["tags"])
                except (json.JSONDecodeError, TypeError):
                    mem["tags"] = []
            if isinstance(mem.get("linked_memories"), str):
                try:
                    mem["linked_memories"] = json.loads(mem["linked_memories"])
                except (json.JSONDecodeError, TypeError):
                    mem["linked_memories"] = []
            memories[mem["id"]] = mem

    return memories


def _compute_bm25_ranks(
    conn: sqlite3.Connection,
    query: str,
    candidate_ids: set[int],
) -> dict[int, int]:
    """
    Run FTS5 search and assign rank positions to candidates.

    Returns a dict mapping memory_id -> 0-based rank position.
    Only candidates present in candidate_ids are included.
    """
    escaped = _escape_fts5_query(query)
    ranks: dict[int, int] = {}
    try:
        rows = conn.execute(
            "SELECT rowid, rank FROM memory_fts "
            "WHERE memory_fts MATCH ? "
            "ORDER BY rank LIMIT 1000",
            (escaped,),
        ).fetchall()
        position = 0
        for row in rows:
            rowid = row[0] if isinstance(row, tuple) else row["rowid"]
            if rowid in candidate_ids:
                ranks[rowid] = position
                position += 1
    except Exception as exc:
        logger.debug("BM25 ranking failed: %s", exc)

    return ranks


def _compute_vector_ranks(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    candidate_ids: set[int],
) -> dict[int, int]:
    """
    Run vector similarity search and assign rank positions to candidates.

    Returns a dict mapping memory_id -> 0-based rank position.
    Only candidates present in candidate_ids are included.
    """
    ranks: dict[int, int] = {}
    try:
        rows = conn.execute(
            "SELECT id, distance FROM memory_vec "
            "WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT 1000",
            (_serialize_f32(query_embedding),),
        ).fetchall()
        position = 0
        for row in rows:
            mid = row[0] if isinstance(row, tuple) else row["id"]
            if mid in candidate_ids:
                ranks[mid] = position
                position += 1
    except Exception as exc:
        logger.debug("Vector ranking failed: %s", exc)

    return ranks


def _rrf_fusion(
    bm25_ranks: dict[int, int],
    vec_ranks: dict[int, int],
    all_ids: set[int],
    k: int = 60,
) -> dict[int, float]:
    """
    Reciprocal Rank Fusion across BM25 and vector rank positions.

    For each candidate:
        rrf_score = 1/(k + bm25_rank) + 1/(k + vec_rank)

    Candidates not found in a particular ranking get a penalty rank of 9999.
    """
    scores: dict[int, float] = {}
    penalty_rank = 9999

    for mid in all_ids:
        bm25_r = bm25_ranks.get(mid, penalty_rank)
        vec_r = vec_ranks.get(mid, penalty_rank)
        scores[mid] = 1.0 / (k + bm25_r) + 1.0 / (k + vec_r)

    return scores


def score_candidates(
    conn: sqlite3.Connection,
    candidate_ids: list[int],
    query: str,
    query_embedding: list[float],
    config: "Config",
) -> list[dict]:
    """
    Stage 2: Fine-grained scoring on the candidate shortlist.

    For each candidate:
      1. Compute BM25 and vector rank positions
      2. Apply RRF fusion
      3. Apply full cognitive scoring (ACT-R + fan effect + confidence + surprise)
      4. Combine into a final score

    Args:
        conn: SQLite connection.
        candidate_ids: Memory IDs from Stage 1.
        query: The raw user query string.
        query_embedding: Pre-computed query embedding vector.
        config: The memory-v3 Config object.

    Returns:
        List of scored memory dicts, sorted by final_score descending.
        Each dict includes hybrid_score, bm25_rank, vec_rank, rrf_score,
        activation, adj_confidence, and final_score.
    """
    if not candidate_ids:
        return []

    # Fetch full records
    memories = _fetch_memories(conn, candidate_ids)
    if not memories:
        return []

    candidate_set = set(memories.keys())
    k = getattr(config, "rrf_k", 60)

    # Compute rank positions from both retrieval channels
    bm25_ranks = _compute_bm25_ranks(conn, query, candidate_set)
    vec_ranks = _compute_vector_ranks(conn, query_embedding, candidate_set)

    # RRF fusion
    rrf_scores = _rrf_fusion(bm25_ranks, vec_ranks, candidate_set, k=k)

    # Build tag fan counts across the candidate set for fan effect
    tag_counter: Counter = Counter()
    for mem in memories.values():
        for tag in (mem.get("tags") or []):
            tag_counter[tag] += 1
    tag_fan: dict[str, int] = dict(tag_counter)

    # Extract context tags from the query (simple: use query words)
    context_tags: set[str] = set(query.lower().split())

    # Score each candidate using the unified scoring pipeline
    from ..scoring import score_memory

    scored_results: list[dict[str, Any]] = []

    for mid, mem in memories.items():
        rrf = rrf_scores.get(mid, 0.0)

        # Attach hybrid_score so score_memory can use it
        mem["hybrid_score"] = rrf
        mem["bm25_rank"] = bm25_ranks.get(mid, 9999)
        mem["vec_rank"] = vec_ranks.get(mid, 9999)
        mem["rrf_score"] = rrf

        # Full cognitive scoring
        final, activation, adj_conf = score_memory(
            mem, rrf, context_tags, tag_fan, config=config,
        )
        mem["final_score"] = final
        mem["activation"] = activation
        mem["adj_confidence"] = adj_conf

        scored_results.append(mem)

    # Sort by final score descending
    scored_results.sort(key=lambda r: r["final_score"], reverse=True)

    logger.debug(
        "Stage 2 scored %d candidates (top score=%.4f)",
        len(scored_results),
        scored_results[0]["final_score"] if scored_results else 0.0,
    )

    return scored_results
