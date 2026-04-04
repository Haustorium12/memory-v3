"""
Tests for memory_v3.scoring.surprise -- Titans-inspired surprise scoring.

Covers compute_surprise and assign_cluster with various centroid configurations.
"""

import numpy as np
import pytest

from memory_v3.scoring.surprise import compute_surprise, assign_cluster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int = 768, seed: int = 0) -> np.ndarray:
    """Generate a deterministic unit vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# compute_surprise
# ---------------------------------------------------------------------------

class TestComputeSurprise:

    def test_compute_surprise_no_centroids(self):
        """With no centroids, surprise should be 0.5 (neutral default)."""
        emb = _unit_vec(seed=42)
        surprise = compute_surprise(emb, [])
        assert abs(surprise - 0.5) < 1e-5

    def test_compute_surprise_identical(self):
        """An embedding identical to a centroid should have surprise ~0.0."""
        emb = _unit_vec(seed=1)
        centroids = [emb.copy()]
        surprise = compute_surprise(emb, centroids)
        assert surprise < 0.05  # near zero

    def test_compute_surprise_novel(self):
        """An embedding orthogonal to all centroids should have high surprise."""
        # Create centroids concentrated in one area
        c1 = np.zeros(768, dtype=np.float32)
        c1[0] = 1.0
        c2 = np.zeros(768, dtype=np.float32)
        c2[1] = 1.0

        # Embedding in a completely different direction
        emb = np.zeros(768, dtype=np.float32)
        emb[500] = 1.0

        surprise = compute_surprise(emb, [c1, c2])
        assert surprise > 0.9  # very novel

    def test_compute_surprise_clamped(self):
        """Surprise should always be in [0, 1]."""
        emb = _unit_vec(seed=10)
        centroids = [_unit_vec(seed=s) for s in range(5)]
        surprise = compute_surprise(emb, centroids)
        assert 0.0 <= surprise <= 1.0


# ---------------------------------------------------------------------------
# assign_cluster
# ---------------------------------------------------------------------------

class TestAssignCluster:

    def test_assign_cluster(self):
        """Should return index of the nearest centroid."""
        emb = _unit_vec(seed=1)
        centroids = [
            _unit_vec(seed=100),  # index 0
            emb + np.random.randn(768).astype(np.float32) * 0.01,  # index 1 -- very close
            _unit_vec(seed=200),  # index 2
        ]
        # Normalize centroid 1 to unit length
        centroids[1] = centroids[1] / np.linalg.norm(centroids[1])

        idx = assign_cluster(emb, centroids)
        assert idx == 1

    def test_assign_cluster_no_centroids(self):
        """With no centroids, should return 0 (seed cluster)."""
        emb = _unit_vec(seed=42)
        idx = assign_cluster(emb, [])
        assert idx == 0
