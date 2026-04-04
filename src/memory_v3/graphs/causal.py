"""
memory-v3 -- Causal Graph Layer
Cause-effect chains, error-fix pairs, decision consequences.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx

from .base import add_edge_to_graph, add_node_to_graph, load_graph, ppr_search, save_graph

NODE_TYPES = ["cause", "effect", "condition", "decision", "error", "fix"]
EDGE_TYPES = ["caused", "enabled", "prevented", "motivated", "triggered_by", "resolved"]

# Edges that point forward in causal direction (from cause toward effect)
_FORWARD_EDGES = {"caused", "enabled", "motivated"}
# Edges that point backward (from effect toward cause)
_BACKWARD_EDGES = {"triggered_by", "resolved"}


class CausalGraph:
    """Manages the causal knowledge layer -- cause-effect relationships."""

    def __init__(self, graph_dir: Path):
        self.path = graph_dir / "causal.pkl"
        self._G: Optional[nx.DiGraph] = None

    @property
    def G(self) -> nx.DiGraph:
        if self._G is None:
            self._G = load_graph(self.path)
        return self._G

    def save(self) -> None:
        if self._G is not None:
            save_graph(self._G, self.path)

    def add_causes(
        self,
        entities: list[dict],
        source_file: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> dict:
        """
        Add causal entities and relationships.

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

            node_type = entity.get("type", "cause")
            if node_type not in NODE_TYPES:
                node_type = "cause"

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

                edge_type = rel.get("edge_type", "caused")
                if edge_type not in EDGE_TYPES:
                    edge_type = "caused"

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
        """Search the causal graph using PPR."""
        return ppr_search(query, self.G, embeddings_module=embeddings_module, top_k=top_k)

    def get_causal_chain(self, start_node: str, max_depth: int = 5) -> list[dict]:
        """
        Trace a causal chain from a start node.

        Uses BFS to follow causal edges in both forward and backward directions,
        building an ordered chain from root cause to final effect.

        Returns a list of dicts representing the chain, each with:
          - node: str
          - depth: int (distance from start_node)
          - edge_type: str (the edge connecting to previous node in chain)
          - direction: str ("forward" or "backward")
          - node_type, and other node attrs
        """
        if start_node not in self.G:
            return []

        visited = {start_node}
        chain = []

        # BFS: explore both successors and predecessors
        queue = deque([(start_node, 0, None, None)])

        while queue:
            current, depth, edge_type, direction = queue.popleft()

            if depth > 0:
                attrs = dict(self.G.nodes[current])
                attrs.pop("embedding", None)
                chain.append({
                    "node": current,
                    "depth": depth,
                    "edge_type": edge_type,
                    "direction": direction,
                    **attrs,
                })

            if depth >= max_depth:
                continue

            # Forward: follow outgoing edges (cause -> effect)
            for successor in self.G.successors(current):
                if successor in visited:
                    continue
                edge_data = self.G.edges[current, successor]
                et = edge_data.get("edge_type", "caused")
                visited.add(successor)
                queue.append((successor, depth + 1, et, "forward"))

            # Backward: follow incoming edges (effect <- cause)
            for predecessor in self.G.predecessors(current):
                if predecessor in visited:
                    continue
                edge_data = self.G.edges[predecessor, current]
                et = edge_data.get("edge_type", "caused")
                visited.add(predecessor)
                queue.append((predecessor, depth + 1, et, "backward"))

        # Sort: backward items (root causes) first by depth descending,
        # then forward items (effects) by depth ascending
        backward = [c for c in chain if c["direction"] == "backward"]
        forward = [c for c in chain if c["direction"] == "forward"]
        backward.sort(key=lambda x: x["depth"], reverse=True)
        forward.sort(key=lambda x: x["depth"])

        return backward + forward

    def stats(self) -> dict:
        """Return statistics about the causal graph."""
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
            "layer": "causal",
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "node_types": type_counts,
            "edge_types": edge_type_counts,
        }
