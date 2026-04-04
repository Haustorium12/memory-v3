"""
memory-v3 -- Temporal Graph Layer
Events, milestones, conversations, and their temporal relationships.
All nodes carry a timestamp attribute for chronological ordering.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx

from .base import add_edge_to_graph, add_node_to_graph, load_graph, ppr_search, save_graph

NODE_TYPES = ["event", "milestone", "conversation", "decision", "memory_ref"]
EDGE_TYPES = ["preceded_by", "followed_by", "concurrent_with", "overlaps_with", "same_session"]


class TemporalGraph:
    """Manages the temporal knowledge layer -- events and their temporal ordering."""

    def __init__(self, graph_dir: Path):
        self.path = graph_dir / "temporal.pkl"
        self._G: Optional[nx.DiGraph] = None

    @property
    def G(self) -> nx.DiGraph:
        if self._G is None:
            self._G = load_graph(self.path)
        return self._G

    def save(self) -> None:
        if self._G is not None:
            save_graph(self._G, self.path)

    def add_events(
        self,
        events: list[dict],
        source_file: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> dict:
        """
        Add temporal events and relationships.

        Each event dict should have:
          - name: str
          - type: str (one of NODE_TYPES)
          - timestamp: str (ISO 8601) -- required for temporal nodes
          - properties: dict (optional)
          - relationships: list[dict] (optional)
            Each: {target, edge_type, description?, confidence?}

        Returns stats: {nodes_added, nodes_updated, edges_added}
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        stats = {"nodes_added": 0, "nodes_updated": 0, "edges_added": 0}

        for event in events:
            name = event.get("name", "")
            if not name:
                continue

            node_type = event.get("type", "event")
            if node_type not in NODE_TYPES:
                node_type = "event"

            properties = event.get("properties", {})
            # Temporal nodes always have a timestamp
            properties["timestamp"] = event.get("timestamp", ts)

            result = add_node_to_graph(
                self.G, name, node_type, properties, source_file, ts
            )
            if result["is_new"]:
                stats["nodes_added"] += 1
            else:
                stats["nodes_updated"] += 1

            for rel in event.get("relationships", []):
                target = rel.get("target", "")
                if not target:
                    continue

                edge_type = rel.get("edge_type", "followed_by")
                if edge_type not in EDGE_TYPES:
                    edge_type = "followed_by"

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
        """Search the temporal graph using PPR."""
        return ppr_search(query, self.G, embeddings_module=embeddings_module, top_k=top_k)

    def get_timeline(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Return events in chronological order, optionally filtered by date range.

        Args:
            start_date: ISO 8601 string, include events on or after this date.
            end_date: ISO 8601 string, include events on or before this date.
            limit: Maximum number of events to return.

        Returns list of dicts with node info, sorted by timestamp ascending.
        """
        events = []
        for node, data in self.G.nodes(data=True):
            node_ts = data.get("timestamp")
            if not node_ts:
                continue

            # Filter by date range
            if start_date and node_ts < start_date:
                continue
            if end_date and node_ts > end_date:
                continue

            attrs = dict(data)
            attrs.pop("embedding", None)
            events.append({"node": node, **attrs})

        # Sort chronologically
        events.sort(key=lambda e: e.get("timestamp", ""))

        return events[:limit]

    def stats(self) -> dict:
        """Return statistics about the temporal graph."""
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
            "layer": "temporal",
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "node_types": type_counts,
            "edge_types": edge_type_counts,
        }
