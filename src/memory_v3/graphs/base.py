"""
memory-v3 -- Graph Base Layer
Shared utilities for graph persistence, node/edge management, and PPR search.
"""

from __future__ import annotations

import pickle
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np


def load_graph(path: Path) -> nx.DiGraph:
    """Load a graph from a pickle file, or return an empty DiGraph."""
    if path.exists():
        with open(path, "rb") as f:
            G = pickle.load(f)
        if not isinstance(G, nx.DiGraph):
            G = nx.DiGraph(G)
        return G
    return nx.DiGraph()


def save_graph(G: nx.DiGraph, path: Path) -> None:
    """Persist a graph to a pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def add_node_to_graph(
    G: nx.DiGraph,
    name: str,
    node_type: str,
    properties: Optional[dict] = None,
    source_file: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> dict:
    """Add or update a node in the graph. Returns stats about the operation."""
    is_new = name not in G
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    attrs = {
        "node_type": node_type,
        "source_file": source_file,
        "created_at": ts if is_new else G.nodes[name].get("created_at", ts),
        "updated_at": ts,
    }
    if properties:
        attrs.update(properties)

    if is_new:
        G.add_node(name, **attrs)
    else:
        G.nodes[name].update(attrs)

    return {"node": name, "is_new": is_new, "total_nodes": G.number_of_nodes()}


def add_edge_to_graph(
    G: nx.DiGraph,
    source: str,
    target: str,
    edge_type: str,
    description: str = "",
    source_file: Optional[str] = None,
    timestamp: Optional[str] = None,
    confidence: float = 0.8,
) -> dict:
    """Add or update an edge in the graph. Returns stats about the operation."""
    is_new = not G.has_edge(source, target)
    ts = timestamp or datetime.now(timezone.utc).isoformat()

    # Ensure both nodes exist
    for node in (source, target):
        if node not in G:
            G.add_node(node, node_type="unknown", created_at=ts, updated_at=ts)

    G.add_edge(
        source,
        target,
        edge_type=edge_type,
        description=description,
        source_file=source_file,
        confidence=confidence,
        created_at=ts if is_new else G.edges[source, target].get("created_at", ts),
        updated_at=ts,
    )

    return {"edge": (source, target), "is_new": is_new, "total_edges": G.number_of_edges()}


def ppr_search(
    query: str,
    G: nx.DiGraph,
    embeddings_module=None,
    top_k: int = 10,
    alpha: float = 0.85,
) -> list[dict]:
    """
    Personalized PageRank search over a graph.

    1. Tokenize query into words.
    2. Find seed nodes whose names contain any query word (case-insensitive).
    3. If embeddings_module is provided and seeds are sparse, boost with embedding similarity.
    4. Run nx.pagerank with personalization biased toward seed nodes.
    5. Return top-K nodes with their PPR scores and metadata.
    """
    if G.number_of_nodes() == 0:
        return []

    query_lower = query.lower()
    query_words = set(query_lower.split())

    # Find seed nodes by name overlap
    personalization = {}
    all_nodes = list(G.nodes)

    for node in all_nodes:
        node_lower = str(node).lower()
        # Score by how many query words appear in the node name
        match_count = sum(1 for w in query_words if w in node_lower)
        if match_count > 0:
            personalization[node] = match_count

        # Also check node_type and description-like attributes
        attrs = G.nodes[node]
        for attr_key in ("node_type", "description", "content"):
            attr_val = str(attrs.get(attr_key, "")).lower()
            word_hits = sum(1 for w in query_words if w in attr_val)
            if word_hits > 0:
                personalization[node] = personalization.get(node, 0) + word_hits * 0.5

    # If we have an embeddings module and few/no seeds, use embedding similarity
    if embeddings_module and len(personalization) < 3:
        try:
            query_emb = np.array(embeddings_module.embed_text(query))
            for node in all_nodes:
                node_emb_data = G.nodes[node].get("embedding")
                if node_emb_data is not None:
                    node_emb = np.array(node_emb_data)
                    sim = float(np.dot(query_emb, node_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(node_emb) + 1e-9
                    ))
                    if sim > 0.3:
                        personalization[node] = personalization.get(node, 0) + sim * 2
        except Exception:
            pass

    # Fallback: uniform personalization if no seeds found
    if not personalization:
        personalization = {n: 1.0 for n in all_nodes}

    # Normalize personalization
    total = sum(personalization.values())
    personalization = {k: v / total for k, v in personalization.items()}

    # Run PPR -- handle disconnected graphs gracefully
    try:
        scores = nx.pagerank(G, alpha=alpha, personalization=personalization, max_iter=100)
    except nx.PowerIterationFailedConvergence:
        scores = nx.pagerank(G, alpha=alpha, personalization=personalization, max_iter=500, tol=1e-4)

    # Sort by score descending, return top_k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for node, score in ranked:
        attrs = dict(G.nodes[node])
        attrs.pop("embedding", None)  # Don't return raw embeddings
        results.append({
            "node": node,
            "score": round(score, 6),
            **attrs,
        })

    return results
