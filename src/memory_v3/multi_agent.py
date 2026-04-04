"""
memory-v3 -- Multi-Agent Authority & Sync

Authority chain, Kafka-style consumer offsets, and conflict detection for
multi-agent memory collaboration.

The authority chain is configurable via ``config.authority_chain`` (populated
from the MEMORY_V3_AUTHORITY_CHAIN env var or sensible defaults).  Lower
numeric levels have higher authority: a writer can only overwrite memories
created by agents at the same level or below.

Conflict detection uses cosine similarity to find near-duplicate memories
from different agents and logs them for human resolution.
"""

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np

from . import db, embeddings
from .config import get_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authority helpers
# ---------------------------------------------------------------------------

def get_authority_level(author: str) -> int:
    """Return the numeric authority level for an agent/author.

    Lower is higher authority.  Unknown authors default to the highest
    numeric level in the chain + 1 (lowest authority).
    """
    cfg = get_config()
    chain = cfg.authority_chain
    if author in chain:
        return chain[author]
    # Unknown author gets lowest authority
    return max(chain.values()) + 1 if chain else 99


def can_overwrite(writer: str, existing_author: str) -> bool:
    """Check if *writer* has sufficient authority to overwrite a memory
    created by *existing_author*.

    A writer can overwrite if their authority level is <= the existing
    author's level (i.e., same or higher authority).
    """
    return get_authority_level(writer) <= get_authority_level(existing_author)


# ---------------------------------------------------------------------------
# Consumer offsets (Kafka-style agent sync)
# ---------------------------------------------------------------------------

def get_offset(conn, agent_id: str) -> dict:
    """Get the current consumer offset for an agent.

    Returns
    -------
    dict
        ``agent_id``, ``last_read_line``, ``last_read_time``.
        If no offset exists, ``last_read_line`` is 0.
    """
    row = conn.execute(
        "SELECT * FROM agent_offsets WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    if row is None:
        return {
            "agent_id": agent_id,
            "last_read_line": 0,
            "last_read_time": None,
        }
    return dict(row)


def set_offset(conn, agent_id: str, last_read_line: int):
    """Update the consumer offset for an agent.

    Creates the row if it doesn't exist (upsert).
    """
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO agent_offsets (agent_id, last_read_line, last_read_time)
        VALUES (?, ?, ?)
        ON CONFLICT(agent_id) DO UPDATE
        SET last_read_line = excluded.last_read_line,
            last_read_time = excluded.last_read_time""",
        (agent_id, last_read_line, now),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Changelog sync
# ---------------------------------------------------------------------------

def get_unread_changelog(agent_id: str, conn, changelog_path: str | None = None) -> dict:
    """Read unread changelog entries for an agent.

    The changelog is a text file where each line is a timestamped event.
    The agent's consumer offset tracks which line was last read.

    Parameters
    ----------
    agent_id : str
        The agent identifier.
    conn : sqlite3.Connection
        Database connection for offset storage.
    changelog_path : str, optional
        Path to the changelog file.  If None, uses
        ``<vault_path>/changelog.md`` from config.

    Returns
    -------
    dict
        ``entries`` (list of unread lines), ``new_offset`` (line number
        after the last entry), ``total_lines``.
    """
    if changelog_path is None:
        cfg = get_config()
        changelog_path = os.path.join(cfg.vault_path, "changelog.md") if cfg.vault_path else ""

    if not changelog_path or not os.path.exists(changelog_path):
        return {"entries": [], "new_offset": 0, "total_lines": 0}

    with open(changelog_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    offset_info = get_offset(conn, agent_id)
    last_read = offset_info["last_read_line"]

    unread = lines[last_read:]
    new_offset = len(lines)

    return {
        "entries": [line.rstrip("\n") for line in unread if line.strip()],
        "new_offset": new_offset,
        "total_lines": len(lines),
    }


def sync_agent(agent_id: str, conn, changelog_path: str | None = None) -> dict:
    """Sync an agent: fetch unread changelog entries and advance offset.

    Combines ``get_unread_changelog`` and ``set_offset`` into a single
    call.  Returns the unread entries and the new offset.
    """
    result = get_unread_changelog(agent_id, conn, changelog_path)
    if result["new_offset"] > 0:
        set_offset(conn, agent_id, result["new_offset"])
    return {
        "agent_id": agent_id,
        "entries_read": len(result["entries"]),
        "entries": result["entries"],
        "new_offset": result["new_offset"],
    }


# ---------------------------------------------------------------------------
# Conflict detection & resolution
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def detect_conflict(
    conn,
    new_content: str,
    new_author: str,
    similar_memories: list[dict],
    threshold: float = 0.85,
) -> dict | None:
    """Detect if a new memory conflicts with existing similar memories.

    A conflict exists when a semantically similar memory was written by a
    *different* agent.  This flags it for human review rather than silently
    overwriting.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    new_content : str
        Content of the proposed new memory.
    new_author : str
        Author/agent writing the new memory.
    similar_memories : list[dict]
        Pre-fetched similar memories (from hybrid search), each with at
        least ``id``, ``content``, ``author``, and optionally a
        ``hybrid_score`` or ``vec_score``.
    threshold : float
        Minimum similarity score to consider a conflict.

    Returns
    -------
    dict or None
        Conflict descriptor if one is found, else None.
    """
    for mem in similar_memories:
        existing_author = mem.get("author", "unknown")

        # Same author -- not a conflict, it's an update
        if existing_author == new_author:
            continue

        # Check similarity score (use hybrid_score if available)
        score = mem.get("hybrid_score", mem.get("vec_score", 0.0))
        if score < threshold:
            continue

        # Found a conflict: different author, high similarity
        return {
            "memory_id_a": mem["id"],
            "memory_id_b": None,  # new memory not yet persisted
            "agent_a": existing_author,
            "agent_b": new_author,
            "description": (
                f"Agent '{new_author}' attempting to write memory similar to "
                f"existing memory #{mem['id']} by '{existing_author}' "
                f"(similarity={score:.3f})"
            ),
            "existing_content": mem["content"],
            "new_content": new_content,
            "similarity": score,
        }

    return None


def log_conflict(conn, conflict: dict) -> int:
    """Persist a detected conflict to the conflicts table.

    Returns the conflict row ID.
    """
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """INSERT INTO conflicts
        (memory_id_a, memory_id_b, agent_a, agent_b, description, created_at)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (
            conflict.get("memory_id_a"),
            conflict.get("memory_id_b"),
            conflict.get("agent_a"),
            conflict.get("agent_b"),
            conflict.get("description", ""),
            now,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_unresolved_conflicts(conn) -> list[dict]:
    """Get all unresolved conflicts, newest first."""
    rows = conn.execute(
        """SELECT * FROM conflicts
        WHERE resolved = 0
        ORDER BY created_at DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def resolve_conflict(
    conn,
    conflict_id: int,
    resolved_by: str = "human",
) -> bool:
    """Mark a conflict as resolved.

    Parameters
    ----------
    conflict_id : int
        Row ID from the conflicts table.
    resolved_by : str
        Who resolved the conflict (default ``"human"``).

    Returns
    -------
    bool
        True if the conflict was found and updated, False otherwise.
    """
    cursor = conn.execute(
        """UPDATE conflicts
        SET resolved = 1, resolved_by = ?
        WHERE id = ? AND resolved = 0""",
        (resolved_by, conflict_id),
    )
    conn.commit()
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Authority-aware write
# ---------------------------------------------------------------------------

def authority_write(
    conn,
    content: str,
    embedding: list[float],
    author: str,
    content_type: str = "fact",
    check_conflicts: bool = True,
    conflict_threshold: float = 0.85,
    **kwargs,
) -> dict:
    """Write a memory with authority checks and optional conflict detection.

    1. Determines the author's authority level.
    2. Optionally searches for similar existing memories.
    3. If a similar memory exists from a higher-authority agent, the write
       is blocked and a conflict is logged.
    4. If a similar memory exists from a same/lower-authority agent, the
       existing memory is superseded.
    5. Otherwise, the memory is added normally.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    content : str
        Memory content text.
    embedding : list[float]
        Pre-computed embedding vector.
    author : str
        Agent/author identifier.
    content_type : str
        Memory content type (fact, episode, decision, etc.).
    check_conflicts : bool
        Whether to run conflict detection (default True).
    conflict_threshold : float
        Similarity threshold for conflict detection.
    **kwargs
        Additional arguments passed to ``db.add_memory()``.

    Returns
    -------
    dict
        Result with ``status`` ("added", "superseded", "blocked", "conflict_logged"),
        ``memory_id`` (if added), and optional ``conflict`` descriptor.
    """
    authority = get_authority_level(author)

    # Search for similar memories if conflict checking is enabled
    if check_conflicts:
        try:
            similar = db.hybrid_search(
                conn, content, embedding, limit=5
            )
        except Exception as exc:
            log.warning("Conflict check search failed: %s", exc)
            similar = []

        if similar:
            conflict = detect_conflict(
                conn, content, author, similar, threshold=conflict_threshold
            )

            if conflict:
                existing_author = conflict["agent_a"]

                # Can we overwrite?
                if can_overwrite(author, existing_author):
                    # Supersede the existing memory
                    existing_id = conflict["memory_id_a"]
                    memory_id = db.add_memory(
                        conn,
                        content=content,
                        embedding=embedding,
                        content_type=content_type,
                        author=author,
                        authority_level=authority,
                        supersedes=existing_id,
                        **kwargs,
                    )
                    return {
                        "status": "superseded",
                        "memory_id": memory_id,
                        "superseded_id": existing_id,
                    }
                else:
                    # Blocked -- log the conflict for human review
                    conflict_id = log_conflict(conn, conflict)
                    return {
                        "status": "conflict_logged",
                        "conflict_id": conflict_id,
                        "conflict": conflict,
                    }

    # No conflicts or conflict checking disabled -- add normally
    memory_id = db.add_memory(
        conn,
        content=content,
        embedding=embedding,
        content_type=content_type,
        author=author,
        authority_level=authority,
        **kwargs,
    )
    return {
        "status": "added",
        "memory_id": memory_id,
    }
