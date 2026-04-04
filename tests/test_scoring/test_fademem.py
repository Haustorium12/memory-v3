"""
Tests for memory_v3.scoring.fademem -- FadeMem importance scoring and archival logic.

Covers the 4-weight importance function, archival decisions, and memory layer classification.
"""

import pytest

from memory_v3.scoring.fademem import (
    importance_score,
    should_archive,
    get_memory_layer,
)


# ---------------------------------------------------------------------------
# Importance Score (4 weights: alpha + beta + gamma + delta)
# ---------------------------------------------------------------------------

class TestImportanceScore:

    def test_importance_4_weights(self):
        """The 4-weight formula should combine relevance, frequency, recency, and surprise."""
        score = importance_score(
            relevance=0.8,
            access_count=10,
            max_access_count=20,
            hours_since_access=1.0,
            surprise_score=0.6,
            decay_rate=0.1,
        )
        # Should be in (0, 1)
        assert 0.0 < score < 1.0

    def test_importance_all_max(self):
        """Perfect scores across all dimensions should yield close to 1.0."""
        score = importance_score(
            relevance=1.0,
            access_count=100,
            max_access_count=100,
            hours_since_access=0.0,
            surprise_score=1.0,
        )
        assert score > 0.9

    def test_importance_all_min(self):
        """Zero across all dimensions should yield close to 0.0."""
        score = importance_score(
            relevance=0.0,
            access_count=0,
            max_access_count=100,
            hours_since_access=10000.0,
            surprise_score=0.0,
        )
        assert score < 0.1

    def test_weights_sum_to_one(self):
        """Default weights alpha+beta+gamma+delta should sum to 1.0."""
        # The defaults: alpha=0.30, beta=0.25, gamma=0.25, delta=0.20
        assert abs(0.30 + 0.25 + 0.25 + 0.20 - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Archival Decision
# ---------------------------------------------------------------------------

class TestShouldArchive:

    def test_should_archive_layer1_never(self):
        """Layer 1 (Constitutional) should NEVER be archived, even with zero importance."""
        result = should_archive(
            importance=0.0,
            age_days=9999,
            governance_layer=1,
        )
        assert result is False

    def test_should_archive_layer2_never(self):
        """Layer 2 (Legislative) should NEVER be archived."""
        result = should_archive(
            importance=0.0,
            age_days=9999,
            governance_layer=2,
        )
        assert result is False

    def test_should_archive_layer4_fast(self):
        """Layer 4 (Ephemeral) with low importance and old age should be archived."""
        result = should_archive(
            importance=0.01,
            age_days=30,
            governance_layer=4,
        )
        assert result is True

    def test_should_archive_layer4_too_young(self):
        """Layer 4 should NOT be archived if younger than min_archive_age_days."""
        result = should_archive(
            importance=0.01,
            age_days=5,  # < 14 days
            governance_layer=4,
        )
        assert result is False

    def test_should_archive_protected_tags_block(self):
        """Protected tags should block archival regardless of score."""
        result = should_archive(
            importance=0.0,
            age_days=999,
            tags={"identity"},
            governance_layer=4,
        )
        assert result is False


# ---------------------------------------------------------------------------
# Memory Layer (STM/LTM/current)
# ---------------------------------------------------------------------------

class TestGetMemoryLayer:

    def test_get_memory_layer_ltm(self):
        """High importance should classify as LTM."""
        assert get_memory_layer(0.8) == "LTM"

    def test_get_memory_layer_stm(self):
        """Medium importance should classify as STM."""
        assert get_memory_layer(0.5) == "STM"

    def test_get_memory_layer_current(self):
        """Low importance should classify as current (demotion candidate)."""
        assert get_memory_layer(0.1) == "current"

    def test_get_memory_layer_boundary_ltm(self):
        """Exactly 0.7 should be LTM."""
        assert get_memory_layer(0.7) == "LTM"

    def test_get_memory_layer_boundary_stm(self):
        """Exactly 0.3 should be STM."""
        assert get_memory_layer(0.3) == "STM"
