"""
memory-v3 -- 4-Phase Sleep Consolidation Cycles

Inspired by biological memory consolidation during sleep:
  1. Replay: Re-process recent memories for cross-connections (zettelkasten)
  2. Reorganize: Update cluster centroids and community assignments
  3. Compress: Merge highly similar memory groups into summaries
  4. Prune: Archive memories below governance-aware thresholds
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

from ..config import Config, get_config
from ..db import (
    add_memory, archive_memory, log_consolidation, _serialize_f32,
)
from ..embeddings import embed_with_cache
from ..scoring.fademem import importance_score, should_archive, get_memory_layer
from ..scoring.surprise import update_centroids
from ..scoring.actr import cosine_similarity


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_consolidation(
    conn: sqlite3.Connection,
    cycle_type: str = "full",
    config: Optional[Config] = None,
) -> dict:
    """Run sleep consolidation.

    cycle_type: 'replay', 'reorganize', 'compress', 'prune', or 'full'.
    Returns dict of per-phase stats.
    """
    cfg = config or get_config()
    stats = {}

    if cycle_type in ("replay", "full"):
        stats["replay"] = _replay(conn, cfg)

    if cycle_type in ("reorganize", "full"):
        stats["reorganize"] = _reorganize(conn, cfg)

    if cycle_type in ("compress", "full"):
        stats["compress"] = _compress(conn, cfg)

    if cycle_type in ("prune", "full"):
        stats["prune"] = _prune(conn, cfg)

    # Log the consolidation cycle
    log_consolidation(conn, cycle_type, stats)

    return stats


def _replay(conn: sqlite3.Connection, config: Config) -> dict:
    """Re-process recent memories for cross-connections.

    Gets all memories created in the last 24 hours and runs
    zettelkasten auto_link() against the full memory set.
    """
    stats = {"memories_processed": 0, "new_links_created": 0}

    if not config.enable_zettelkasten:
        return stats

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

    rows = conn.execute(
        """SELECT id FROM memories
        WHERE archived = 0 AND created_at > ?
        ORDER BY created_at DESC""",
        (cutoff,),
    ).fetchall()

    if not rows:
        return stats

    from ..graphs.zettelkasten import auto_link

    for row in rows:
        mid = row[0] if not isinstance(row, dict) else row["id"]

        # Get the embedding for this memory from vec table
        vec_row = conn.execute(
            "SELECT embedding FROM memory_vec WHERE id = ?", (mid,)
        ).fetchone()

        if vec_row is None:
            continue

        emb_blob = vec_row[0] if not isinstance(vec_row, dict) else vec_row["embedding"]
        embedding = np.frombuffer(emb_blob, dtype=np.float32)

        try:
            link_stats = auto_link(
                conn, mid, embedding,
                threshold=config.zettel_link_threshold,
            )
            stats["memories_processed"] += 1
            stats["new_links_created"] += link_stats.get("links_created", 0)
        except Exception:
            continue

    return stats


def _reorganize(conn: sqlite3.Connection, config: Config) -> dict:
    """Update cluster centroids and community assignments.

    Fetches all non-archived embeddings, groups by current cluster_id,
    recomputes centroids, and reassigns memories to nearest centroid.
    """
    stats = {"clusters_updated": 0, "memories_reassigned": 0}

    # Fetch all active memory embeddings
    rows = conn.execute(
        """SELECT m.id, m.cluster_id FROM memories m
        WHERE m.archived = 0"""
    ).fetchall()

    if not rows:
        return stats

    # Load embeddings from vec table
    embeddings_by_cluster: dict[int, list[np.ndarray]] = defaultdict(list)
    memory_embeddings: dict[int, np.ndarray] = {}
    memory_clusters: dict[int, int] = {}

    for row in rows:
        mid = row[0] if not isinstance(row, dict) else row["id"]
        cluster_id = row[1] if not isinstance(row, dict) else row["cluster_id"]
        cluster_id = cluster_id or 0

        vec_row = conn.execute(
            "SELECT embedding FROM memory_vec WHERE id = ?", (mid,)
        ).fetchone()

        if vec_row is None:
            continue

        emb_blob = vec_row[0] if not isinstance(vec_row, dict) else vec_row["embedding"]
        emb = np.frombuffer(emb_blob, dtype=np.float32).copy()

        embeddings_by_cluster[cluster_id].append(emb)
        memory_embeddings[mid] = emb
        memory_clusters[mid] = cluster_id

    if not memory_embeddings:
        return stats

    # If we have very few memories, use a single cluster
    num_memories = len(memory_embeddings)
    if num_memories < 5:
        # Put everything in cluster 0
        all_embs = list(memory_embeddings.values())
        update_centroids(conn, {0: all_embs})
        stats["clusters_updated"] = 1
        return stats

    # Determine number of clusters (sqrt heuristic, min 2, max 50)
    import math
    k = max(2, min(50, int(math.sqrt(num_memories))))

    # Simple k-means clustering (limited iterations)
    all_ids = list(memory_embeddings.keys())
    all_embs = np.stack([memory_embeddings[mid] for mid in all_ids])

    # Initialize centroids from random existing embeddings
    rng = np.random.default_rng(42)
    init_indices = rng.choice(len(all_ids), size=min(k, len(all_ids)), replace=False)
    centroids = all_embs[init_indices].copy()

    # Normalize centroids
    for i in range(len(centroids)):
        norm = np.linalg.norm(centroids[i])
        if norm > 0:
            centroids[i] = centroids[i] / norm

    assignments = np.zeros(len(all_ids), dtype=int)

    for iteration in range(20):
        # Assign each memory to nearest centroid
        new_assignments = np.zeros(len(all_ids), dtype=int)
        for i, emb in enumerate(all_embs):
            best_sim = -1.0
            best_c = 0
            for c_idx, centroid in enumerate(centroids):
                sim = cosine_similarity(emb, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_c = c_idx
            new_assignments[i] = best_c

        # Check convergence
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments

        # Recompute centroids
        for c_idx in range(len(centroids)):
            mask = assignments == c_idx
            if mask.sum() > 0:
                centroids[c_idx] = all_embs[mask].mean(axis=0)
                norm = np.linalg.norm(centroids[c_idx])
                if norm > 0:
                    centroids[c_idx] = centroids[c_idx] / norm

    # Persist new centroids
    new_embeddings_by_cluster: dict[int, list[np.ndarray]] = defaultdict(list)
    for i, mid in enumerate(all_ids):
        cluster_id = int(assignments[i])
        new_embeddings_by_cluster[cluster_id].append(all_embs[i])

    update_centroids(conn, new_embeddings_by_cluster)
    stats["clusters_updated"] = len(new_embeddings_by_cluster)

    # Update cluster_id on memories
    for i, mid in enumerate(all_ids):
        new_cluster = int(assignments[i])
        old_cluster = memory_clusters.get(mid, -1)
        if new_cluster != old_cluster:
            conn.execute(
                "UPDATE memories SET cluster_id = ?, updated_at = ? WHERE id = ?",
                (new_cluster, _now_iso(), mid),
            )
            stats["memories_reassigned"] += 1

    conn.commit()
    return stats


def _compress(conn: sqlite3.Connection, config: Config) -> dict:
    """Merge highly similar memory groups.

    Finds groups of 3+ memories with pairwise cosine > MERGE_THRESHOLD.
    For each group, generates a merged summary via LLM, creates a new
    consolidated memory, and archives the originals.
    """
    stats = {"groups_found": 0, "memories_merged": 0, "new_memories_created": 0}

    merge_threshold = config.consolidation_merge_threshold
    min_group = config.consolidation_min_group_size

    # Get all active memories with embeddings
    rows = conn.execute(
        """SELECT id, content, content_type, governance_layer, protected,
                  tags, source_file, author, confidence
        FROM memories
        WHERE archived = 0 AND protected = 0
          AND governance_layer IN (3, 4)
        ORDER BY created_at DESC
        LIMIT 500"""
    ).fetchall()

    if len(rows) < min_group:
        return stats

    # Load embeddings
    mem_data = {}
    mem_embs = {}
    for row in rows:
        mem = dict(row)
        mid = mem["id"]

        vec_row = conn.execute(
            "SELECT embedding FROM memory_vec WHERE id = ?", (mid,)
        ).fetchone()
        if vec_row is None:
            continue

        emb_blob = vec_row[0] if not isinstance(vec_row, dict) else vec_row["embedding"]
        mem_embs[mid] = np.frombuffer(emb_blob, dtype=np.float32).copy()
        mem_data[mid] = mem

    if len(mem_embs) < min_group:
        return stats

    # Find groups using greedy clustering
    ids = list(mem_embs.keys())
    used = set()
    groups = []

    for i, mid_a in enumerate(ids):
        if mid_a in used:
            continue

        group = [mid_a]
        for j in range(i + 1, len(ids)):
            mid_b = ids[j]
            if mid_b in used:
                continue

            # Check if mid_b is similar to ALL members of the group
            all_similar = True
            for gid in group:
                sim = cosine_similarity(mem_embs[gid], mem_embs[mid_b])
                if sim < merge_threshold:
                    all_similar = False
                    break

            if all_similar:
                group.append(mid_b)

        if len(group) >= min_group:
            groups.append(group)
            used.update(group)

    stats["groups_found"] = len(groups)

    if not groups:
        return stats

    # Merge each group
    import ollama
    model = config.llm_model

    for group in groups:
        contents = [mem_data[mid]["content"] for mid in group]
        combined = "\n---\n".join(contents)

        prompt = (
            "Merge these related memories into a single concise statement "
            "that preserves all key facts:\n\n"
            f"{combined}\n\n"
            "Merged memory:"
        )

        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 512},
            )
            merged_content = response["message"]["content"].strip()
        except Exception:
            # On LLM failure, just concatenate with separator
            merged_content = " | ".join(contents)

        # Create merged memory
        merged_embedding = embed_with_cache(merged_content)
        first_mem = mem_data[group[0]]

        new_id = add_memory(
            conn,
            content=merged_content,
            embedding=merged_embedding,
            content_type=first_mem.get("content_type", "fact"),
            source_file=first_mem.get("source_file"),
            author=first_mem.get("author", "consolidation"),
            confidence=max(mem_data[mid].get("confidence", 0.5) for mid in group),
            governance_layer=first_mem.get("governance_layer", 3),
            structure_type="consolidated",
        )

        # Archive originals
        for mid in group:
            archive_memory(conn, mid, reason="consolidated")
            stats["memories_merged"] += 1

        stats["new_memories_created"] += 1

    return stats


def _prune(conn: sqlite3.Connection, config: Config) -> dict:
    """Archive memories below threshold per governance layer.

    Uses scoring.fademem.should_archive() with governance-aware thresholds.
    """
    stats = {"memories_pruned": 0, "by_layer": {}}

    # Get max access count for normalization
    max_ac_row = conn.execute(
        "SELECT MAX(access_count) FROM memories WHERE archived = 0"
    ).fetchone()
    max_access_count = (max_ac_row[0] or 1) if max_ac_row else 1

    now = datetime.now(timezone.utc)

    # Check archivable layers (3 and 4)
    for layer_id in (3, 4):
        layer_stats = 0
        rows = conn.execute(
            """SELECT id, content, importance_score, access_count,
                      last_accessed_at, created_at, tags, governance_layer,
                      surprise_score, decay_rate
            FROM memories
            WHERE archived = 0 AND protected = 0
              AND governance_layer = ?""",
            (layer_id,),
        ).fetchall()

        for row in rows:
            mem = dict(row)
            mid = mem["id"]

            # Parse tags
            try:
                tags_raw = mem.get("tags", "[]")
                if isinstance(tags_raw, str):
                    tags = set(json.loads(tags_raw))
                else:
                    tags = set(tags_raw or [])
            except (json.JSONDecodeError, TypeError):
                tags = set()

            # Compute age in days
            try:
                created = datetime.fromisoformat(mem["created_at"].replace("Z", "+00:00"))
                age_days = (now - created).total_seconds() / 86400
            except (ValueError, TypeError):
                age_days = 0

            # Compute hours since last access
            try:
                last_access = datetime.fromisoformat(
                    mem["last_accessed_at"].replace("Z", "+00:00")
                )
                hours_since = (now - last_access).total_seconds() / 3600
            except (ValueError, TypeError):
                hours_since = age_days * 24

            # Recompute importance
            imp = importance_score(
                relevance=0.3,  # Base relevance for pruning context
                access_count=mem.get("access_count", 1),
                max_access_count=max_access_count,
                hours_since_access=hours_since,
                surprise_score=mem.get("surprise_score", 0.5),
                decay_rate=mem.get("decay_rate", 0.1),
            )

            # Check archive criteria
            if should_archive(imp, age_days, tags=tags, governance_layer=layer_id):
                archive_memory(conn, mid, reason="prune_consolidation")
                layer_stats += 1

                # Also update memory layer classification
                new_layer = get_memory_layer(imp)
                if new_layer == "current":
                    # Already archiving
                    pass

        stats["by_layer"][layer_id] = layer_stats
        stats["memories_pruned"] += layer_stats

    return stats
