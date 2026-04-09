"""Titans-inspired surprise scoring for memory consolidation.

Surprise measures how *unexpected* a new memory is relative to existing
knowledge clusters.  High-surprise memories carry novel information and
receive a scoring boost to ensure they are retained and surfaced.

Uses centroid-based clustering:  each cluster is represented by the mean
embedding of its members, stored in the ``cluster_centroids`` table.
"""

from __future__ import annotations

import json
import sqlite3
from typing import List, Optional, Tuple

import numpy as np

from .actr import cosine_similarity


def compute_surprise(
    embedding: np.ndarray,
    centroids: List[np.ndarray],
) -> float:
    """Compute surprise as distance from the nearest cluster centroid.

    surprise = 1.0 - max(cosine_similarity(embedding, c) for c in centroids)

    Returns 0.5 (neutral) when no centroids exist.
    """
    if not centroids or len(centroids) == 0:
        return 0.5

    embedding = np.asarray(embedding, dtype=np.float32).ravel()
    if np.linalg.norm(embedding) == 0:
        return 0.5

    max_sim = -1.0
    for centroid in centroids:
        c = np.asarray(centroid, dtype=np.float32).ravel()
        sim = cosine_similarity(embedding, c)
        if sim > max_sim:
            max_sim = sim

    # Clamp to [0, 1]
    surprise = 1.0 - max_sim
    return max(min(surprise, 1.0), 0.0)


def assign_cluster(
    embedding: np.ndarray,
    centroids: List[np.ndarray],
) -> int:
    """Return the index of the nearest centroid.

    If no centroids exist, returns 0 (the embedding would seed cluster 0).
    """
    if not centroids or len(centroids) == 0:
        return 0

    embedding = np.asarray(embedding, dtype=np.float32).ravel()
    best_idx = 0
    best_sim = -1.0

    for idx, centroid in enumerate(centroids):
        c = np.asarray(centroid, dtype=np.float32).ravel()
        sim = cosine_similarity(embedding, c)
        if sim > best_sim:
            best_sim = sim
            best_idx = idx

    return best_idx


def update_centroids(
    conn: sqlite3.Connection,
    embeddings_by_cluster: dict[int, List[np.ndarray]],
) -> None:
    """Recompute centroids as the mean of member embeddings and persist.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection with a ``cluster_centroids`` table.
    embeddings_by_cluster : dict
        Mapping of cluster_id -> list of member embedding arrays.
    """
    cursor = conn.cursor()

    # Ensure the table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cluster_centroids (
            cluster_id INTEGER PRIMARY KEY,
            centroid_embedding TEXT NOT NULL,
            member_count INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    for cluster_id, embeddings in embeddings_by_cluster.items():
        if not embeddings:
            continue

        # Stack and compute mean
        stacked = np.stack([np.asarray(e, dtype=np.float32).ravel() for e in embeddings])
        centroid = np.mean(stacked, axis=0)

        # Normalize the centroid for consistent cosine comparisons
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        centroid_json = json.dumps(centroid.tolist())

        cursor.execute(
            """
            INSERT INTO cluster_centroids (cluster_id, centroid_embedding, member_count, created_at, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(cluster_id) DO UPDATE SET
                centroid_embedding = excluded.centroid_embedding,
                member_count = excluded.member_count,
                updated_at = CURRENT_TIMESTAMP
            """,
            (cluster_id, centroid_json, len(embeddings)),
        )

    conn.commit()


def get_centroids(conn: sqlite3.Connection) -> List[Tuple[int, np.ndarray]]:
    """Read all cluster centroids from the database.

    Returns a list of (cluster_id, centroid_array) tuples.
    """
    cursor = conn.cursor()

    # Gracefully handle missing table
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='cluster_centroids'"
    )
    if cursor.fetchone() is None:
        return []

    cursor.execute("SELECT cluster_id, centroid_embedding FROM cluster_centroids")
    results = []
    for row in cursor.fetchall():
        cluster_id = row[0]
        try:
            embedding = np.array(json.loads(row[1]), dtype=np.float32)
            results.append((cluster_id, embedding))
        except (json.JSONDecodeError, TypeError):
            continue

    return results
