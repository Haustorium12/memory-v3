"""
memory-v3 -- Semantic Graph Layer
Concepts, tools, projects, patterns, and their conceptual relationships.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx

from .base import add_edge_to_graph, add_node_to_graph, load_graph, ppr_search, save_graph

NODE_TYPES = ["concept", "project", "tool", "topic", "pattern", "memory_ref"]
EDGE_TYPES = ["related_to", "uses", "part_of", "implements", "depends_on", "similar_to"]


class SemanticGraph:
    """Manages the semantic knowledge layer -- concepts and their relationships."""

    def __init__(self, graph_dir: Path):
        self.path = graph_dir / "semantic.pkl"
        self._G: Optional[nx.DiGraph] = None

    @property
    def G(self) -> nx.DiGraph:
        if self._G is None:
            self._G = load_graph(self.path)
        return self._G

    def save(self) -> None:
        if self._G is not None:
            save_graph(self._G, self.path)

    def add_entities(
        self,
        entities: list[dict],
        source_file: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> dict:
        """
        Add entities and relationships to the semantic graph.

        Each entity dict should have:
          - name: str
          - type: str (one of NODE_TYPES)
          - properties: dict (optional)

        If an entity dict has 'relationships', each should be:
          - target: str
          - edge_type: str (one of EDGE_TYPES)
          - description: str (optional)
          - confidence: float (optional, default 0.8)

        Returns stats: {nodes_added, nodes_updated, edges_added}
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        stats = {"nodes_added": 0, "nodes_updated": 0, "edges_added": 0}

        for entity in entities:
            name = entity.get("name", "")
            if not name:
                continue

            node_type = entity.get("type", "concept")
            if node_type not in NODE_TYPES:
                node_type = "concept"

            properties = entity.get("properties", {})
            result = add_node_to_graph(
                self.G, name, node_type, properties, source_file, ts
            )
            if result["is_new"]:
                stats["nodes_added"] += 1
            else:
                stats["nodes_updated"] += 1

            # Process relationships attached to this entity
            for rel in entity.get("relationships", []):
                target = rel.get("target", "")
                if not target:
                    continue

                edge_type = rel.get("edge_type", "related_to")
                if edge_type not in EDGE_TYPES:
                    edge_type = "related_to"

                edge_result = add_edge_to_graph(
                    self.G,
                    name,
                    target,
                    edge_type,
                    description=rel.get("description", ""),
                    source_file=source_file,
                    timestamp=ts,
                    confidence=rel.get("confidence", 0.8),
                )
                if edge_result["is_new"]:
                    stats["edges_added"] += 1

        return stats

    def search(self, query: str, top_k: int = 10, embeddings_module=None) -> list[dict]:
        """Search the semantic graph using PPR."""
        return ppr_search(query, self.G, embeddings_module=embeddings_module, top_k=top_k)

    def get_related(self, node_name: str, max_depth: int = 2) -> list[dict]:
        """
        Get all nodes related to a given node up to max_depth hops.
        Returns list of dicts with node info, depth, and connecting edge type.
        """
        if node_name not in self.G:
            return []

        visited = {node_name}
        results = []
        queue = [(node_name, 0)]

        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            # Check both successors and predecessors for full connectivity
            for neighbor in set(self.G.successors(current)) | set(self.G.predecessors(current)):
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                # Determine edge info
                if self.G.has_edge(current, neighbor):
                    edge_data = self.G.edges[current, neighbor]
                    direction = "outgoing"
                elif self.G.has_edge(neighbor, current):
                    edge_data = self.G.edges[neighbor, current]
                    direction = "incoming"
                else:
                    edge_data = {}
                    direction = "unknown"

                attrs = dict(self.G.nodes[neighbor])
                attrs.pop("embedding", None)

                results.append({
                    "node": neighbor,
                    "depth": depth + 1,
                    "edge_type": edge_data.get("edge_type", "unknown"),
                    "direction": direction,
                    "confidence": edge_data.get("confidence", 0.0),
                    **attrs,
                })

                queue.append((neighbor, depth + 1))

        return results

    def stats(self) -> dict:
        """Return statistics about the semantic graph."""
        G = self.G
        type_counts = {}
        for _, data in G.nodes(data=True):
            nt = data.get("node_type", "unknown")
            type_counts[nt] = type_counts.get(nt, 0) + 1

        edge_type_counts = {}
        for _, _, data in G.edges(data=True):
            et = data.get("edge_type", "unknown")
            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1

        return {
            "layer": "semantic",
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "node_types": type_counts,
            "edge_types": edge_type_counts,
        }
