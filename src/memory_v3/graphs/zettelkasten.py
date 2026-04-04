"""
memory-v3 -- Zettelkasten / A-MEM Self-Linking
Automatic bidirectional linking of memories based on embedding similarity.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def auto_link(
    conn: sqlite3.Connection,
    memory_id: int,
    embedding: list[float] | np.ndarray,
    threshold: float = 0.5,
    max_links: int = 10,
    semantic_graph=None,
) -> dict:
    """
    Automatically link a memory to its most similar neighbors.

    1. Query all other memories' embeddings from the database.
    2. Compute cosine similarity with the given embedding.
    3. For each neighbor above threshold (up to max_links):
       a. Create a bidirectional link in linked_memories JSON on both memories.
       b. Optionally add a "similar_to" edge in the semantic graph.
    4. Return stats.

    Args:
        conn: SQLite connection. Expects a 'memories' table with columns:
              id, content, linked_memories (JSON text), embedding (BLOB or JSON).
        memory_id: The memory to auto-link.
        embedding: The embedding vector for this memory.
        threshold: Minimum cosine similarity to create a link.
        max_links: Maximum number of links to create.
        semantic_graph: Optional SemanticGraph instance to add "similar_to" edges.

    Returns:
        {"links_created": int, "memories_updated": int}
    """
    query_vec = np.array(embedding, dtype=np.float32)
    stats = {"links_created": 0, "memories_updated": 0}

    # Get all other memories with embeddings
    cursor = conn.execute(
        "SELECT id, content, linked_memories FROM memories WHERE id != ?",
        (memory_id,),
    )
    rows = cursor.fetchall()

    if not rows:
        return stats

    # Also need embeddings -- check if there's an embedding column or vec table
    # Try the vec_memories virtual table first, then fall back to embedding column
    candidate_embeddings = {}
    try:
        vec_cursor = conn.execute(
            "SELECT rowid, embedding FROM vec_memories WHERE rowid != ?",
            (memory_id,),
        )
        for rowid, emb_blob in vec_cursor:
            candidate_embeddings[rowid] = np.frombuffer(emb_blob, dtype=np.float32)
    except Exception:
        # No vec_memories table; try embedding column on memories table
        try:
            emb_cursor = conn.execute(
                "SELECT id, embedding FROM memories WHERE id != ? AND embedding IS NOT NULL",
                (memory_id,),
            )
            for mid, emb_data in emb_cursor:
                if isinstance(emb_data, bytes):
                    candidate_embeddings[mid] = np.frombuffer(emb_data, dtype=np.float32)
                elif isinstance(emb_data, str):
                    candidate_embeddings[mid] = np.array(json.loads(emb_data), dtype=np.float32)
        except Exception:
            pass

    if not candidate_embeddings:
        return stats

    # Compute similarities and rank
    similarities = []
    for mid, vec in candidate_embeddings.items():
        sim = _cosine_similarity(query_vec, vec)
        if sim >= threshold:
            similarities.append((mid, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarities[:max_links]

    if not top_matches:
        return stats

    # Get current linked_memories for the source
    source_row = conn.execute(
        "SELECT linked_memories FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    if source_row is None:
        return stats

    source_links = json.loads(source_row[0]) if source_row[0] else []
    source_link_ids = {link.get("memory_id") for link in source_links if isinstance(link, dict)}

    ts = datetime.now(timezone.utc).isoformat()
    updated_memory_ids = set()

    for target_id, sim in top_matches:
        # Skip if already linked
        if target_id in source_link_ids:
            continue

        # Get target's current links
        target_row = conn.execute(
            "SELECT content, linked_memories FROM memories WHERE id = ?", (target_id,)
        ).fetchone()
        if target_row is None:
            continue

        target_content = target_row[0] or ""
        target_links = json.loads(target_row[1]) if target_row[1] else []

        # Add forward link: source -> target
        source_links.append({
            "memory_id": target_id,
            "similarity": round(sim, 4),
            "linked_at": ts,
            "description": f"Auto-linked (cosine={sim:.3f})",
        })

        # Add reverse link: target -> source
        target_link_ids = {l.get("memory_id") for l in target_links if isinstance(l, dict)}
        if memory_id not in target_link_ids:
            target_links.append({
                "memory_id": memory_id,
                "similarity": round(sim, 4),
                "linked_at": ts,
                "description": f"Auto-linked (cosine={sim:.3f})",
            })
            conn.execute(
                "UPDATE memories SET linked_memories = ? WHERE id = ?",
                (json.dumps(target_links), target_id),
            )
            updated_memory_ids.add(target_id)

        # Add edge to semantic graph if available
        if semantic_graph is not None:
            from .base import add_edge_to_graph
            add_edge_to_graph(
                semantic_graph.G,
                f"memory:{memory_id}",
                f"memory:{target_id}",
                "similar_to",
                description=f"Auto-linked (cosine={sim:.3f})",
                confidence=sim,
                timestamp=ts,
            )

        stats["links_created"] += 1

    # Update source memory's linked_memories
    conn.execute(
        "UPDATE memories SET linked_memories = ? WHERE id = ?",
        (json.dumps(source_links), memory_id),
    )
    updated_memory_ids.add(memory_id)
    conn.commit()

    stats["memories_updated"] = len(updated_memory_ids)
    return stats


def get_links(conn: sqlite3.Connection, memory_id: int) -> list[dict]:
    """
    Get all linked memories for a given memory.

    Returns list of dicts with:
      - memory_id: int
      - content: str (first 200 chars)
      - similarity: float
      - linked_at: str
      - description: str
    """
    row = conn.execute(
        "SELECT linked_memories FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()

    if row is None or not row[0]:
        return []

    links = json.loads(row[0])
    results = []

    for link in links:
        if not isinstance(link, dict):
            continue

        linked_id = link.get("memory_id")
        if linked_id is None:
            continue

        # Fetch the linked memory's content
        linked_row = conn.execute(
            "SELECT content FROM memories WHERE id = ?", (linked_id,)
        ).fetchone()
        content_preview = ""
        if linked_row and linked_row[0]:
            content_preview = linked_row[0][:200]

        results.append({
            "memory_id": linked_id,
            "content": content_preview,
            "similarity": link.get("similarity", 0.0),
            "linked_at": link.get("linked_at", ""),
            "description": link.get("description", ""),
        })

    return results


def add_manual_link(
    conn: sqlite3.Connection,
    memory_id_a: int,
    memory_id_b: int,
    description: str = "",
) -> bool:
    """
    Create an explicit bidirectional link between two memories.

    Returns True if the link was created, False if either memory doesn't exist
    or they are already linked.
    """
    # Verify both memories exist
    row_a = conn.execute(
        "SELECT linked_memories FROM memories WHERE id = ?", (memory_id_a,)
    ).fetchone()
    row_b = conn.execute(
        "SELECT linked_memories FROM memories WHERE id = ?", (memory_id_b,)
    ).fetchone()

    if row_a is None or row_b is None:
        return False

    ts = datetime.now(timezone.utc).isoformat()
    link_desc = description or "Manual link"

    # Update A -> B
    links_a = json.loads(row_a[0]) if row_a[0] else []
    existing_ids_a = {l.get("memory_id") for l in links_a if isinstance(l, dict)}
    if memory_id_b not in existing_ids_a:
        links_a.append({
            "memory_id": memory_id_b,
            "similarity": 1.0,
            "linked_at": ts,
            "description": link_desc,
        })
        conn.execute(
            "UPDATE memories SET linked_memories = ? WHERE id = ?",
            (json.dumps(links_a), memory_id_a),
        )

    # Update B -> A
    links_b = json.loads(row_b[0]) if row_b[0] else []
    existing_ids_b = {l.get("memory_id") for l in links_b if isinstance(l, dict)}
    if memory_id_a not in existing_ids_b:
        links_b.append({
            "memory_id": memory_id_a,
            "similarity": 1.0,
            "linked_at": ts,
            "description": link_desc,
        })
        conn.execute(
            "UPDATE memories SET linked_memories = ? WHERE id = ?",
            (json.dumps(links_b), memory_id_b),
        )

    conn.commit()
    return True
