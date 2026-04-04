"""
memory-v3 -- Stage 3: Post-Retrieval Organization
Uses graph context, structure tags, and zettelkasten links to organize
and annotate results for LLM consumption.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)


def _dominant_structure_type(results: list[dict]) -> str:
    """
    Determine the dominant structure_type among results by majority vote.

    Falls back to "narrative" if no clear winner.
    """
    if not results:
        return "narrative"

    counts: Counter = Counter()
    for r in results:
        st = r.get("structure_type", "narrative")
        counts[st] += 1

    best, best_count = counts.most_common(1)[0]
    # Only use the detected type if it covers at least 40% of results
    if best_count >= len(results) * 0.4:
        return best
    return "narrative"


def _sort_temporal(results: list[dict]) -> list[dict]:
    """
    Sort results chronologically and add sequence_position markers.
    """
    results.sort(key=lambda r: r.get("created_at", ""))
    for i, r in enumerate(results):
        r["sequence_position"] = i + 1
        r["sequence_total"] = len(results)
    return results


def _sort_causal(results: list[dict], conn: sqlite3.Connection) -> list[dict]:
    """
    Try to order results along a causal chain using the causal graph.

    If the causal graph is available and contains relevant nodes, orders
    results from root cause to final effect. Falls back to chronological
    ordering if causal chain construction fails.
    """
    try:
        from ..graphs import GraphManager

        gm = GraphManager()
        causal = gm._get_layer("causal")

        # Map memory IDs to causal graph node names
        # Memories reference nodes via content keywords
        memory_nodes: dict[int, str] = {}
        for r in results:
            mid = r.get("id")
            content = r.get("content", "")
            # Check if any graph node name appears in the memory content
            for node in causal.G.nodes:
                if str(node).lower() in content.lower():
                    memory_nodes[mid] = str(node)
                    break

        if len(memory_nodes) < 2:
            # Not enough causal connections; fall back to chronological
            return _sort_temporal(results)

        # Get causal chain from the first found node
        start_node = next(iter(memory_nodes.values()))
        chain = causal.get_causal_chain(start_node, max_depth=10)
        chain_order = {entry["node"]: i for i, entry in enumerate(chain)}

        # Sort memories by their position in the causal chain
        def causal_sort_key(r):
            mid = r.get("id")
            node = memory_nodes.get(mid)
            if node and node in chain_order:
                return chain_order[node]
            return 9999

        results.sort(key=causal_sort_key)

        for i, r in enumerate(results):
            mid = r.get("id")
            node = memory_nodes.get(mid)
            if node and node in chain_order:
                r["causal_position"] = chain_order[node]
                r["causal_node"] = node

        return results

    except Exception as exc:
        logger.debug("Causal ordering failed, falling back to temporal: %s", exc)
        return _sort_temporal(results)


def _group_by_entity(results: list[dict], conn: sqlite3.Connection) -> list[dict]:
    """
    Group results by entity graph relationships.

    Looks for entity names in memory content and groups memories that
    share entity connections. Adds entity_group to each result.
    """
    try:
        from ..graphs import GraphManager

        gm = GraphManager()
        entity_graph = gm._get_layer("entity")

        # Find entity associations for each memory
        entity_groups: dict[str, list[dict]] = {}

        for r in results:
            content = r.get("content", "").lower()
            matched_entity = None

            for node in entity_graph.G.nodes:
                if str(node).lower() in content:
                    matched_entity = str(node)
                    break

            if matched_entity:
                r["entity_group"] = matched_entity
                entity_groups.setdefault(matched_entity, []).append(r)
            else:
                r["entity_group"] = "_ungrouped"
                entity_groups.setdefault("_ungrouped", []).append(r)

        # Flatten groups: grouped entities first, ungrouped last
        ordered = []
        for entity_name, group in sorted(entity_groups.items()):
            if entity_name == "_ungrouped":
                continue
            # Sort within group by final_score
            group.sort(key=lambda r: r.get("final_score", 0), reverse=True)
            ordered.extend(group)

        # Append ungrouped at the end
        ungrouped = entity_groups.get("_ungrouped", [])
        ungrouped.sort(key=lambda r: r.get("final_score", 0), reverse=True)
        ordered.extend(ungrouped)

        return ordered

    except Exception as exc:
        logger.debug("Entity grouping failed: %s", exc)
        return results


def _add_graph_neighbors(
    results: list[dict],
    conn: sqlite3.Connection,
    layers: list[str],
) -> None:
    """
    For each result, query graph layers for 1-hop neighbors and attach
    as a "related_nodes" list.
    """
    try:
        from ..graphs import GraphManager

        gm = GraphManager()

        for r in results:
            related: list[dict] = []
            content = r.get("content", "").lower()

            for layer_name in layers:
                try:
                    graph_layer = gm._get_layer(layer_name)
                    G = graph_layer.G

                    # Find nodes that match this memory's content
                    for node in G.nodes:
                        if str(node).lower() in content:
                            # Get 1-hop neighbors
                            for neighbor in G.successors(node):
                                edge_data = G.edges[node, neighbor]
                                related.append({
                                    "node": str(neighbor),
                                    "layer": layer_name,
                                    "edge_type": edge_data.get("edge_type", "related_to"),
                                    "direction": "outgoing",
                                })
                            for predecessor in G.predecessors(node):
                                edge_data = G.edges[predecessor, node]
                                related.append({
                                    "node": str(predecessor),
                                    "layer": layer_name,
                                    "edge_type": edge_data.get("edge_type", "related_to"),
                                    "direction": "incoming",
                                })
                            break  # One node match per layer is enough
                except Exception:
                    continue

            # Deduplicate by node name
            seen_nodes: set[str] = set()
            unique_related: list[dict] = []
            for item in related:
                if item["node"] not in seen_nodes:
                    seen_nodes.add(item["node"])
                    unique_related.append(item)

            r["related_nodes"] = unique_related[:10]  # Cap at 10 neighbors

    except Exception as exc:
        logger.debug("Graph neighbor lookup failed: %s", exc)
        for r in results:
            r.setdefault("related_nodes", [])


def _add_zettelkasten_links(
    results: list[dict],
    conn: sqlite3.Connection,
) -> None:
    """
    Add zettelkasten linked_memories as a "see_also" field on each result.
    """
    from ..graphs.zettelkasten import get_links

    for r in results:
        mid = r.get("id")
        if mid is None:
            r["see_also"] = []
            continue

        try:
            links = get_links(conn, mid)
            r["see_also"] = [
                {
                    "memory_id": link["memory_id"],
                    "similarity": link.get("similarity", 0.0),
                    "content_preview": link.get("content", "")[:100],
                }
                for link in links[:5]  # Cap at 5 see-also links
            ]
        except Exception:
            r["see_also"] = []


def organize(
    conn: sqlite3.Connection,
    results: list[dict],
    query: str,
    config: "Config",
) -> list[dict]:
    """
    Stage 3: Post-retrieval organization and annotation.

    1. Detect dominant structure type among results
    2. Route query to graph layers
    3. Apply structure-appropriate ordering (temporal, causal, entity)
    4. Annotate each result with graph neighbors and zettelkasten links
    5. Generate a structure_hint for LLM consumption

    Args:
        conn: SQLite connection.
        results: Scored results from Stage 2.
        query: The raw user query string.
        config: The memory-v3 Config object.

    Returns:
        Organized and annotated result list.
    """
    if not results:
        return []

    # Determine dominant structure type
    enable_tags = getattr(config, "enable_structure_tags", True)
    if enable_tags:
        structure_type = _dominant_structure_type(results)
    else:
        structure_type = "narrative"

    # Route query to determine relevant graph layers
    from ..graphs.router import route_query
    layers = route_query(query)

    # Apply structure-appropriate ordering
    if structure_type == "timeline" or "temporal" in layers:
        results = _sort_temporal(results)
    elif "causal" in layers:
        results = _sort_causal(results, conn)
    elif "entity" in layers:
        results = _group_by_entity(results, conn)
    # narrative / ledger / taxonomy: keep score-based ordering from Stage 2

    # Annotate with graph neighbors
    if getattr(config, "enable_multigraph", True):
        _add_graph_neighbors(results, conn, layers)

    # Annotate with zettelkasten links
    if getattr(config, "enable_zettelkasten", True):
        _add_zettelkasten_links(results, conn)

    # Generate structure hint
    from .structure_tags import get_structure_hint
    hint = get_structure_hint(structure_type, len(results))

    # Attach metadata to each result
    for r in results:
        r["structure_hint"] = hint
        r["detected_structure"] = structure_type
        r["routed_layers"] = layers

    logger.debug(
        "Stage 3 organized %d results (structure=%s, layers=%s)",
        len(results), structure_type, layers,
    )

    return results
