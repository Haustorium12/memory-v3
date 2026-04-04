"""
Tests for memory_v3.graphs.router -- query routing to graph layers.

Covers temporal, causal, entity, semantic, and multi-layer routing.
"""

import pytest

from memory_v3.graphs.router import route_query


# ---------------------------------------------------------------------------
# Single-Layer Routing
# ---------------------------------------------------------------------------

class TestSingleLayerRouting:

    def test_route_temporal(self):
        """'when did X happen' should route to temporal."""
        layers = route_query("when did the deployment happen")
        assert "temporal" in layers

    def test_route_causal(self):
        """'why did X fail' should route to causal."""
        layers = route_query("why did the build fail")
        assert "causal" in layers

    def test_route_entity(self):
        """'who built X' should route to entity."""
        layers = route_query("who built the memory system")
        assert "entity" in layers

    def test_route_default_semantic(self):
        """A generic query with no signal words should default to semantic."""
        layers = route_query("tell me about memory systems")
        assert "semantic" in layers

    def test_route_about_is_semantic(self):
        """'about' is a semantic signal word."""
        layers = route_query("what is this about")
        assert "semantic" in layers


# ---------------------------------------------------------------------------
# Multi-Layer Routing
# ---------------------------------------------------------------------------

class TestMultiLayerRouting:

    def test_route_multi_temporal_causal(self):
        """'when and why did X happen' should route to both temporal and causal."""
        layers = route_query("when and why did the outage happen")
        assert "temporal" in layers
        assert "causal" in layers

    def test_route_multi_entity_temporal(self):
        """'who built X and when' should route to entity and temporal."""
        layers = route_query("who built this and when did they finish")
        assert "entity" in layers
        assert "temporal" in layers


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestRouterEdgeCases:

    def test_empty_query(self):
        """Empty query should default to semantic."""
        layers = route_query("")
        assert layers == ["semantic"]

    def test_case_insensitive(self):
        """Routing should be case-insensitive."""
        layers = route_query("WHEN did this HAPPEN")
        assert "temporal" in layers

    def test_multi_word_signal(self):
        """Multi-word signals like 'led to' should be detected as substrings."""
        layers = route_query("what led to the failure")
        assert "causal" in layers
