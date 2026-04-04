"""
Tests for memory_v3.scoring.hierarchy -- constitutional memory governance layers.

Covers layer configuration, archival policies, and automatic classification.
"""

import pytest

from memory_v3.scoring.hierarchy import (
    LAYER_CONFIGS,
    get_layer_config,
    classify_governance_layer,
)


# ---------------------------------------------------------------------------
# Layer Configuration
# ---------------------------------------------------------------------------

class TestLayerConfigs:

    def test_layer_configs_exist(self):
        """All 4 governance layers should be defined."""
        assert 1 in LAYER_CONFIGS
        assert 2 in LAYER_CONFIGS
        assert 3 in LAYER_CONFIGS
        assert 4 in LAYER_CONFIGS

    def test_layer_1_constitutional(self):
        cfg = LAYER_CONFIGS[1]
        assert cfg.name == "Constitutional"
        assert cfg.decay_rate == 0.0
        assert cfg.can_archive is False

    def test_layer_2_legislative(self):
        cfg = LAYER_CONFIGS[2]
        assert cfg.name == "Legislative"
        assert cfg.can_archive is False

    def test_layer_3_factual(self):
        cfg = LAYER_CONFIGS[3]
        assert cfg.name == "Factual"
        assert cfg.can_archive is True
        assert cfg.min_archive_age_days == 30

    def test_layer_4_ephemeral(self):
        cfg = LAYER_CONFIGS[4]
        assert cfg.name == "Ephemeral"
        assert cfg.can_archive is True
        assert cfg.min_archive_age_days == 14


# ---------------------------------------------------------------------------
# Archival Rules (via layer config, not fademem)
# ---------------------------------------------------------------------------

class TestArchivalPolicy:

    def test_constitutional_no_archive(self):
        """Layer 1 (Constitutional) should never allow archival."""
        cfg = get_layer_config(1)
        assert cfg.can_archive is False

    def test_ephemeral_fast_archive(self):
        """Layer 4 (Ephemeral) should allow archival after 14 days."""
        cfg = get_layer_config(4)
        assert cfg.can_archive is True
        assert cfg.archive_threshold is not None
        assert cfg.min_archive_age_days == 14

    def test_unknown_layer_defaults_to_factual(self):
        """Unknown layer numbers should fall back to Layer 3 (Factual)."""
        cfg = get_layer_config(999)
        assert cfg.name == "Factual"
        assert cfg.level == 3


# ---------------------------------------------------------------------------
# Automatic Classification
# ---------------------------------------------------------------------------

class TestClassifyGovernanceLayer:

    def test_classify_identity_as_constitutional(self):
        """content_type='identity' should classify as Layer 1."""
        layer = classify_governance_layer("identity")
        assert layer == 1

    def test_classify_decision_as_legislative(self):
        """content_type='decision' should classify as Layer 2."""
        layer = classify_governance_layer("decision")
        assert layer == 2

    def test_classify_episode_as_ephemeral(self):
        """content_type='episode' should classify as Layer 4."""
        layer = classify_governance_layer("episode")
        assert layer == 4

    def test_classify_fact_as_factual(self):
        """content_type='fact' should classify as Layer 3."""
        layer = classify_governance_layer("fact")
        assert layer == 3

    def test_classify_protected_bumps_to_legislative(self):
        """A protected memory with generic type gets at least Layer 2."""
        layer = classify_governance_layer("unknown_type", protected=True)
        assert layer == 2

    def test_classify_tag_override(self):
        """Tags should override content_type classification."""
        # identity tag should force Layer 1 even for a 'fact' type
        layer = classify_governance_layer("fact", tags={"identity"})
        assert layer == 1

    def test_classify_decision_tag(self):
        """Decision tag should force Layer 2."""
        layer = classify_governance_layer("fact", tags={"decision"})
        assert layer == 2

    def test_classify_default(self):
        """Unknown content types with no special tags default to Layer 3."""
        layer = classify_governance_layer("widget")
        assert layer == 3
