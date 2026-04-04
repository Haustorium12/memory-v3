"""
Tests for memory_v3.scoring.actr -- ACT-R activation scoring.

Covers cosine similarity, base-level activation, spreading activation,
and retrieval probability.
"""

import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from memory_v3.scoring.actr import (
    cosine_similarity,
    base_level_activation,
    spreading_activation,
    retrieval_probability,
)


# ---------------------------------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:

    def test_cosine_similarity_identical(self):
        """Identical vectors should yield cosine similarity ~1.0."""
        vec = np.random.rand(768).astype(np.float32)
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should yield cosine similarity ~0.0."""
        a = np.zeros(768, dtype=np.float32)
        b = np.zeros(768, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        sim = cosine_similarity(a, b)
        assert abs(sim) < 1e-5

    def test_cosine_similarity_opposite(self):
        """Opposite vectors should yield cosine similarity ~-1.0."""
        vec = np.random.rand(768).astype(np.float32)
        sim = cosine_similarity(vec, -vec)
        assert abs(sim - (-1.0)) < 1e-5

    def test_cosine_similarity_zero_vector(self):
        """A zero vector should yield 0.0 (handled gracefully)."""
        zero = np.zeros(768, dtype=np.float32)
        other = np.ones(768, dtype=np.float32)
        sim = cosine_similarity(zero, other)
        assert sim == 0.0


# ---------------------------------------------------------------------------
# Base-Level Activation
# ---------------------------------------------------------------------------

class TestBaseLevelActivation:

    def test_base_level_activation_positive(self):
        """A recently created, frequently accessed memory should have positive activation."""
        now = datetime.now(timezone.utc)
        created = now - timedelta(hours=1)
        last_accessed = now - timedelta(minutes=5)

        B = base_level_activation(
            access_count=10,
            created_at=created,
            last_accessed_at=last_accessed,
        )
        assert B > 0.0

    def test_base_level_activation_old_memory(self):
        """An old, rarely accessed memory should have lower activation."""
        now = datetime.now(timezone.utc)
        created = now - timedelta(days=365)
        last_accessed = now - timedelta(days=30)

        B_old = base_level_activation(
            access_count=2,
            created_at=created,
            last_accessed_at=last_accessed,
        )
        B_new = base_level_activation(
            access_count=10,
            created_at=now - timedelta(hours=1),
            last_accessed_at=now - timedelta(minutes=5),
        )
        assert B_old < B_new


# ---------------------------------------------------------------------------
# Spreading Activation
# ---------------------------------------------------------------------------

class TestSpreadingActivation:

    def test_spreading_activation_overlap(self):
        """Shared tags should produce positive spreading activation."""
        memory_tags = {"python", "memory", "ai"}
        context_tags = {"python", "ai", "database"}
        tag_fan = {"python": 5, "ai": 3, "database": 2, "memory": 1}

        S = spreading_activation(memory_tags, context_tags, tag_fan)
        assert S > 0.0

    def test_spreading_activation_no_overlap(self):
        """Disjoint tags should produce 0.0 spreading activation."""
        memory_tags = {"python", "memory"}
        context_tags = {"rust", "database"}
        tag_fan = {"python": 5, "memory": 1, "rust": 2, "database": 3}

        S = spreading_activation(memory_tags, context_tags, tag_fan)
        assert S == 0.0

    def test_spreading_activation_empty_context(self):
        """Empty context tags should produce 0.0."""
        S = spreading_activation({"python"}, set(), {"python": 1})
        assert S == 0.0

    def test_spreading_activation_empty_memory(self):
        """Empty memory tags should produce 0.0."""
        S = spreading_activation(set(), {"python"}, {"python": 1})
        assert S == 0.0


# ---------------------------------------------------------------------------
# Retrieval Probability
# ---------------------------------------------------------------------------

class TestRetrievalProbability:

    def test_retrieval_probability_high(self):
        """High activation should yield retrieval probability close to 1.0."""
        P = retrieval_probability(activation=5.0)
        assert P > 0.99

    def test_retrieval_probability_low(self):
        """Very low activation should yield retrieval probability close to 0.0."""
        P = retrieval_probability(activation=-10.0)
        assert P < 0.01

    def test_retrieval_probability_at_threshold(self):
        """At the threshold (tau), P should be ~0.5."""
        P = retrieval_probability(activation=-0.5, tau=-0.5)
        assert abs(P - 0.5) < 0.01
