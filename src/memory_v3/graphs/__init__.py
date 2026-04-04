"""
memory-v3 -- Graph Subpackage
Four-layer knowledge graph: semantic, temporal, causal, entity.
Managed by GraphManager with lazy loading, query routing, and persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import networkx as nx

from .causal import CausalGraph
from .entity import EntityGraph
from .router import route_query
from .semantic import SemanticGraph
from .temporal import TemporalGraph

# Maps edge types to their target graph layer
EDGE_TO_GRAPH = {
    # Semantic
    "related_to": "semantic",
    "uses": "semantic",
    "part_of": "semantic",
    "implements": "semantic",
    "depends_on": "semantic",
    "similar_to": "semantic",
    # Temporal
    "preceded_by": "temporal",
    "followed_by": "temporal",
    "concurrent_with": "temporal",
    "overlaps_with": "temporal",
    "same_session": "temporal",
    # Causal
    "caused": "causal",
    "enabled": "causal",
    "prevented": "causal",
    "motivated": "causal",
    "triggered_by": "causal",
    "resolved": "causal",
    # Entity
    "built": "entity",
    "decided": "entity",
    "maintains": "entity",
    "member_of": "entity",
    "responsible_for": "entity",
}

# Default graph directory
_DEFAULT_GRAPH_DIR = Path.home() / ".memory-v3" / "graphs"

# Layer name -> class mapping
_LAYER_CLASSES = {
    "semantic": SemanticGraph,
    "temporal": TemporalGraph,
    "causal": CausalGraph,
    "entity": EntityGraph,
}


class GraphManager:
    """
    Central manager for all four graph layers.

    Provides lazy loading, unified add/query interface, and persistence.
    """

    def __init__(self, graph_dir=None):
        # Try to get config if available; fall back to default
        resolved_dir = graph_dir
        if resolved_dir is None:
            try:
                from ..config import get_config
                cfg = get_config()
                resolved_dir = getattr(cfg, "GRAPH_DIR", None)
            except (ImportError, AttributeError):
                pass

        self.graph_dir = Path(resolved_dir or _DEFAULT_GRAPH_DIR)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self._graphs: dict[str, SemanticGraph | TemporalGraph | CausalGraph | EntityGraph] = {}

    def _get_layer(self, layer: str):
        """Lazy-load and return a graph layer instance."""
        if layer not in _LAYER_CLASSES:
            raise ValueError(f"Unknown graph layer: {layer}. Must be one of {list(_LAYER_CLASSES)}")

        if layer not in self._graphs:
            cls = _LAYER_CLASSES[layer]
            self._graphs[layer] = cls(self.graph_dir)

        return self._graphs[layer]

    def get_graph(self, layer: str) -> nx.DiGraph:
        """Lazy-load a graph layer and return its underlying nx.DiGraph."""
        return self._get_layer(layer).G

    def add_extracted(
        self,
        entities: list[dict],
        relationships: list[dict],
        source_file: str,
        timestamp: Optional[str] = None,
    ) -> dict:
        """
        Route extracted entities and relationships to the correct graph layers.

        Args:
            entities: List of entity dicts, each with:
                - name: str
                - type: str
                - properties: dict (optional)
                - layer: str (optional, override auto-routing)
            relationships: List of relationship dicts, each with:
                - source: str
                - target: str
                - edge_type: str (used to determine target layer via EDGE_TO_GRAPH)
                - description: str (optional)
                - confidence: float (optional)
            source_file: The source file these were extracted from.
            timestamp: Optional ISO timestamp.

        Returns dict with per-layer stats.
        """
        from datetime import datetime, timezone
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        # Group entities by layer
        layer_entities: dict[str, list[dict]] = {
            "semantic": [], "temporal": [], "causal": [], "entity": [],
        }

        for entity in entities:
            # Determine layer from explicit field or entity type heuristics
            layer = entity.get("layer")
            if not layer:
                etype = entity.get("type", "").lower()
                if etype in ("event", "milestone", "conversation"):
                    layer = "temporal"
                elif etype in ("cause", "effect", "condition", "error", "fix"):
                    layer = "causal"
                elif etype in ("person", "team", "organization", "role"):
                    layer = "entity"
                else:
                    layer = "semantic"

            if layer in layer_entities:
                layer_entities[layer].append(entity)

        # Group relationships by layer using EDGE_TO_GRAPH
        layer_rels: dict[str, list[dict]] = {
            "semantic": [], "temporal": [], "causal": [], "entity": [],
        }

        for rel in relationships:
            edge_type = rel.get("edge_type", "related_to")
            layer = EDGE_TO_GRAPH.get(edge_type, "semantic")
            if layer in layer_rels:
                layer_rels[layer].append(rel)

        stats = {}

        # Process each layer
        for layer_name in _LAYER_CLASSES:
            ents = layer_entities.get(layer_name, [])
            rels = layer_rels.get(layer_name, [])

            if not ents and not rels:
                continue

            graph_layer = self._get_layer(layer_name)

            # Add entities using the layer's add method
            layer_stats = {"nodes_added": 0, "nodes_updated": 0, "edges_added": 0}

            if ents:
                if layer_name == "semantic":
                    result = graph_layer.add_entities(ents, source_file, ts)
                elif layer_name == "temporal":
                    result = graph_layer.add_events(ents, source_file, ts)
                elif layer_name == "causal":
                    result = graph_layer.add_causes(ents, source_file, ts)
                elif layer_name == "entity":
                    result = graph_layer.add_entities(ents, source_file, ts)
                else:
                    result = {}
                layer_stats.update(result)

            # Add standalone relationships (not attached to entities)
            if rels:
                from .base import add_edge_to_graph, add_node_to_graph
                G = graph_layer.G
                for rel in rels:
                    source = rel.get("source", "")
                    target = rel.get("target", "")
                    if not source or not target:
                        continue

                    edge_result = add_edge_to_graph(
                        G,
                        source,
                        target,
                        rel.get("edge_type", "related_to"),
                        description=rel.get("description", ""),
                        source_file=source_file,
                        timestamp=ts,
                        confidence=rel.get("confidence", 0.8),
                    )
                    if edge_result["is_new"]:
                        layer_stats["edges_added"] += 1

            stats[layer_name] = layer_stats

        return stats

    def query(
        self,
        query: str,
        layers: Optional[list[str]] = None,
        top_k: int = 10,
        embeddings_module=None,
    ) -> list[dict]:
        """
        Query across specified graph layers, merge and deduplicate results.

        If layers is None, uses the router to auto-detect which layers to search.

        Returns top_k results sorted by PPR score, with a 'layer' field added.
        """
        if layers is None:
            layers = route_query(query)

        all_results = []

        for layer_name in layers:
            if layer_name not in _LAYER_CLASSES:
                continue

            graph_layer = self._get_layer(layer_name)
            results = graph_layer.search(
                query, top_k=top_k, embeddings_module=embeddings_module
            )

            # Tag each result with its source layer
            for r in results:
                r["layer"] = layer_name

            all_results.extend(results)

        # Deduplicate by node name, keeping highest score
        seen = {}
        for r in all_results:
            node = r["node"]
            if node not in seen or r["score"] > seen[node]["score"]:
                seen[node] = r

        # Sort by score descending and return top_k
        merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    def save_all(self) -> None:
        """Persist all loaded graphs to pickle files."""
        for graph_layer in self._graphs.values():
            graph_layer.save()

    def stats(self) -> dict:
        """Get stats for each graph layer (only loads if already loaded or has data on disk)."""
        result = {}
        for layer_name in _LAYER_CLASSES:
            graph_layer = self._get_layer(layer_name)
            result[layer_name] = graph_layer.stats()

        layer_stats = list(result.values())
        result["total_nodes"] = sum(s["nodes"] for s in layer_stats)
        result["total_edges"] = sum(s["edges"] for s in layer_stats)
        return result
