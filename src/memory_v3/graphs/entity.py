"""
memory-v3 -- Entity Graph Layer
People, teams, projects, organizations, and their operational relationships.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx

from .base import add_edge_to_graph, add_node_to_graph, load_graph, ppr_search, save_graph

NODE_TYPES = ["person", "team", "project", "organization", "tool", "role"]
EDGE_TYPES = ["built", "decided", "maintains", "uses", "member_of", "responsible_for"]


class EntityGraph:
    """Manages the entity knowledge layer -- people, teams, and their relationships."""

    def __init__(self, graph_dir: Path):
        self.path = graph_dir / "entity.pkl"
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
        Add entity nodes and relationships.

        Each entity dict should have:
          - name: str
          - type: str (one of NODE_TYPES)
          - properties: dict (optional)
          - relationships: list[dict] (optional)
            Each: {target, edge_type, description?, confidence?}

        Returns stats: {nodes_added, nodes_updated, edges_added}
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        stats = {"nodes_added": 0, "nodes_updated": 0, "edges_added": 0}

        for entity in entities:
            name = entity.get("name", "")
            if not name:
                continue

            node_type = entity.get("type", "person")
            if node_type not in NODE_TYPES:
                node_type = "person"

            properties = entity.get("properties", {})
            result = add_node_to_graph(
                self.G, name, node_type, properties, source_file, ts
            )
            if result["is_new"]:
                stats["nodes_added"] += 1
            else:
                stats["nodes_updated"] += 1

            for rel in entity.get("relationships", []):
                target = rel.get("target", "")
                if not target:
                    continue

                edge_type = rel.get("edge_type", "built")
                if edge_type not in EDGE_TYPES:
                    edge_type = "built"

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
        """Search the entity graph using PPR."""
        return ppr_search(query, self.G, embeddings_module=embeddings_module, top_k=top_k)

    def get_entity_relations(self, entity_name: str) -> list[dict]:
        """
        Get all direct relationships for a specific entity.

        Returns a list of dicts, each describing a relationship:
          - node: str (the related entity)
          - edge_type: str
          - direction: "outgoing" or "incoming"
          - description: str
          - confidence: float
          - node_type: str (type of the related entity)
        """
        if entity_name not in self.G:
            return []

        relations = []

        # Outgoing edges: this entity -> other
        for target in self.G.successors(entity_name):
            edge_data = self.G.edges[entity_name, target]
            target_attrs = dict(self.G.nodes[target])
            target_attrs.pop("embedding", None)

            relations.append({
                "node": target,
                "edge_type": edge_data.get("edge_type", "unknown"),
                "direction": "outgoing",
                "description": edge_data.get("description", ""),
                "confidence": edge_data.get("confidence", 0.0),
                "node_type": target_attrs.get("node_type", "unknown"),
                "source_file": edge_data.get("source_file"),
            })

        # Incoming edges: other -> this entity
        for source in self.G.predecessors(entity_name):
            edge_data = self.G.edges[source, entity_name]
            source_attrs = dict(self.G.nodes[source])
            source_attrs.pop("embedding", None)

            relations.append({
                "node": source,
                "edge_type": edge_data.get("edge_type", "unknown"),
                "direction": "incoming",
                "description": edge_data.get("description", ""),
                "confidence": edge_data.get("confidence", 0.0),
                "node_type": source_attrs.get("node_type", "unknown"),
                "source_file": edge_data.get("source_file"),
            })

        return relations

    def stats(self) -> dict:
        """Return statistics about the entity graph."""
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
            "layer": "entity",
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "node_types": type_counts,
            "edge_types": edge_type_counts,
        }
