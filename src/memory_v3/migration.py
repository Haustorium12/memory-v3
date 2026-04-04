"""
memory-v3 -- Migration Module
Handles v2 -> v3 database migration and graph migration.
Run via: memory-v3-migrate (pyproject.toml entry point)
"""

import json
import os
import pickle
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from .db import _now_iso


# ---------------------------------------------------------------------------
# v2 Detection
# ---------------------------------------------------------------------------

def detect_v2_db(conn: sqlite3.Connection) -> bool:
    """Check if a database is v2 (has memories table but lacks governance_layer column)."""
    try:
        cols = conn.execute("PRAGMA table_info(memories)").fetchall()
    except Exception:
        return False

    if not cols:
        return False  # No memories table at all

    col_names = {row[1] for row in cols}

    # v2 has the memories table but not governance_layer
    has_memories = "content" in col_names and "activation_score" in col_names
    lacks_v3 = "governance_layer" not in col_names

    return has_memories and lacks_v3


# ---------------------------------------------------------------------------
# v2 -> v3 Database Migration
# ---------------------------------------------------------------------------

# The 12 new columns with their ALTER TABLE statements
V3_NEW_COLUMNS = [
    ("governance_layer", "INTEGER DEFAULT 3"),
    ("confidence_floor", "REAL DEFAULT 0.0"),
    ("last_verified_at", "TEXT"),
    ("verification_status", "TEXT DEFAULT 'unverified'"),
    ("cross_ref_count", "INTEGER DEFAULT 0"),
    ("structure_type", "TEXT DEFAULT 'narrative'"),
    ("surprise_score", "REAL DEFAULT 0.5"),
    ("memory_layer", "TEXT DEFAULT 'STM'"),
    ("cluster_id", "INTEGER"),
    ("linked_memories", "TEXT DEFAULT '[]'"),
    ("link_descriptions", "TEXT DEFAULT '{}'"),
    ("source_hash", "TEXT"),
]

# New v3 tables
V3_NEW_TABLES_SQL = """
-- v3: Sensory buffer (pre-encoding gate)
CREATE TABLE IF NOT EXISTS sensory_buffer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content_type TEXT DEFAULT 'fact',
    source TEXT,
    author TEXT DEFAULT 'claude-code',
    topic_cluster TEXT,
    novelty_score REAL DEFAULT 0.5,
    received_at TEXT NOT NULL,
    processed INTEGER DEFAULT 0,
    processed_at TEXT,
    decision TEXT,
    decision_reason TEXT
);

-- v3: Cluster centroids for memory consolidation
CREATE TABLE IF NOT EXISTS cluster_centroids (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL UNIQUE,
    centroid_embedding BLOB,
    member_count INTEGER DEFAULT 0,
    label TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- v3: Action log (extraction decisions: ADD/UPDATE/DELETE/NONE)
CREATE TABLE IF NOT EXISTS action_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    new_fact TEXT NOT NULL,
    similar_ids TEXT DEFAULT '[]',
    action TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    model_version TEXT
);

-- v3: Consolidation log (merge/split/prune cycles)
CREATE TABLE IF NOT EXISTS consolidation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cycle_type TEXT NOT NULL,
    stats TEXT NOT NULL
);

-- v3: Async write queue
CREATE TABLE IF NOT EXISTS write_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL,
    payload TEXT NOT NULL,
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'pending',
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

-- v3: Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    migrated_at TEXT NOT NULL,
    notes TEXT
);

-- v3 Indexes
CREATE INDEX IF NOT EXISTS idx_memories_governance ON memories(governance_layer);
CREATE INDEX IF NOT EXISTS idx_memories_memory_layer ON memories(memory_layer);
CREATE INDEX IF NOT EXISTS idx_memories_cluster ON memories(cluster_id);
CREATE INDEX IF NOT EXISTS idx_memories_verification ON memories(verification_status);
CREATE INDEX IF NOT EXISTS idx_sensory_processed ON sensory_buffer(processed);
CREATE INDEX IF NOT EXISTS idx_write_queue_status ON write_queue(status, priority);
CREATE INDEX IF NOT EXISTS idx_action_log_action ON action_log(action);
"""


def migrate_v2_to_v3(conn: sqlite3.Connection) -> dict:
    """
    Migrate a v2 database to v3 in-place.

    Steps:
    1. ALTER TABLE memories ADD COLUMN for each of 12 new columns
    2. Classify existing memories into governance layers
    3. CREATE all 6 new tables
    4. Add new indexes
    5. INSERT INTO schema_version
    6. Return stats dict

    Returns:
        dict with migration statistics
    """
    now = _now_iso()
    stats = {
        "columns_added": 0,
        "tables_created": 6,
        "constitutional_count": 0,
        "legislative_count": 0,
        "factual_count": 0,
        "ephemeral_count": 0,
        "migrated_at": now,
    }

    # Step 1: Add new columns
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}

    for col_name, col_def in V3_NEW_COLUMNS:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_def}")
            stats["columns_added"] += 1

    # Step 2: Classify into governance layers
    # Layer 1 - Constitutional: protected memories
    cursor = conn.execute(
        "UPDATE memories SET governance_layer = 1 WHERE protected = 1"
    )
    stats["constitutional_count"] = cursor.rowcount

    # Layer 2 - Legislative: decisions and corrections (not protected)
    cursor = conn.execute(
        "UPDATE memories SET governance_layer = 2 "
        "WHERE content_type IN ('decision', 'correction') AND protected = 0"
    )
    stats["legislative_count"] = cursor.rowcount

    # Layer 4 - Ephemeral: episodes (not protected)
    cursor = conn.execute(
        "UPDATE memories SET governance_layer = 4 "
        "WHERE content_type = 'episode' AND protected = 0"
    )
    stats["ephemeral_count"] = cursor.rowcount

    # Layer 3 - Factual: everything else (already default)
    stats["factual_count"] = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE governance_layer = 3"
    ).fetchone()[0]

    # Step 3: Create new tables and indexes
    conn.executescript(V3_NEW_TABLES_SQL)

    # Step 4: Record schema version
    conn.execute(
        "INSERT INTO schema_version (version, migrated_at, notes) VALUES (?, ?, ?)",
        (3, now, f"Migrated from v2. {stats['columns_added']} columns added.")
    )

    conn.commit()
    return stats


# ---------------------------------------------------------------------------
# Graph Migration (v2 single pickle -> v3 multi-graph pickles)
# ---------------------------------------------------------------------------

# Edge type to graph layer mapping for v3 multigraph
EDGE_TYPE_TO_GRAPH = {
    # Semantic graph (conceptual relationships)
    "semantic": "semantic",
    "related_to": "semantic",
    "similar_to": "semantic",
    "concept": "semantic",

    # Temporal graph (time-based relationships)
    "temporal": "temporal",
    "follows": "temporal",
    "precedes": "temporal",
    "co_occurred": "temporal",
    "same_session": "temporal",

    # Causal graph (cause-effect relationships)
    "causal": "causal",
    "causes": "causal",
    "caused_by": "causal",
    "depends_on": "causal",
    "enables": "causal",
    "blocks": "causal",

    # Structural graph (organizational relationships)
    "structural": "structural",
    "part_of": "structural",
    "contains": "structural",
    "references": "structural",
    "defined_in": "structural",
    "imported_by": "structural",
}

# Default graph for unmapped edge types
DEFAULT_GRAPH = "semantic"


def migrate_graph_v2_to_v3(
    graph_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """
    Migrate a single v2 knowledge graph pickle into 4 separate v3 graph pickles.

    The v2 graph is a single NetworkX graph. v3 uses 4 separate graphs:
    - semantic.pickle: conceptual relationships
    - temporal.pickle: time-based relationships
    - causal.pickle: cause-effect relationships
    - structural.pickle: organizational relationships

    Args:
        graph_path: Path to the v2 graph pickle file
        output_dir: Directory to write the 4 v3 graph pickle files

    Returns:
        dict with migration statistics
    """
    import networkx as nx

    graph_path = Path(graph_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "source": str(graph_path),
        "output_dir": str(output_dir),
        "v2_nodes": 0,
        "v2_edges": 0,
        "graphs_created": {},
        "unmapped_edge_types": [],
    }

    # Load v2 graph
    if not graph_path.exists():
        stats["error"] = f"Graph file not found: {graph_path}"
        return stats

    with open(graph_path, "rb") as f:
        v2_graph = pickle.load(f)

    stats["v2_nodes"] = v2_graph.number_of_nodes()
    stats["v2_edges"] = v2_graph.number_of_edges()

    # Create 4 empty graphs, each inheriting all nodes from v2
    graph_names = ["semantic", "temporal", "causal", "structural"]
    v3_graphs: dict[str, nx.Graph] = {}

    for name in graph_names:
        g = nx.Graph()
        # Copy all nodes with their attributes
        for node, attrs in v2_graph.nodes(data=True):
            g.add_node(node, **attrs)
        v3_graphs[name] = g

    # Route edges to appropriate graphs
    unmapped_types = set()
    for u, v, data in v2_graph.edges(data=True):
        edge_type = data.get("type", data.get("relation", "semantic"))
        edge_type_lower = str(edge_type).lower().replace(" ", "_")

        target_graph = EDGE_TYPE_TO_GRAPH.get(edge_type_lower, DEFAULT_GRAPH)
        if edge_type_lower not in EDGE_TYPE_TO_GRAPH:
            unmapped_types.add(edge_type_lower)

        v3_graphs[target_graph].add_edge(u, v, **data)

    stats["unmapped_edge_types"] = sorted(unmapped_types)

    # Save each graph
    for name, graph in v3_graphs.items():
        out_path = output_dir / f"{name}.pickle"
        with open(out_path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        stats["graphs_created"][name] = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "path": str(out_path),
        }

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for memory-v3-migrate."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate memory-v2 database and graphs to v3 format"
    )
    parser.add_argument(
        "--db", type=str,
        default=os.environ.get("MEMORY_V2_DB",
                               str(Path.home() / ".memory-v2" / "memory.db")),
        help="Path to v2 database (default: ~/.memory-v2/memory.db)"
    )
    parser.add_argument(
        "--graph", type=str, default=None,
        help="Path to v2 knowledge graph pickle (optional)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.environ.get("MEMORY_V3_GRAPH_DIR",
                               str(Path.home() / ".memory-v3" / "graphs")),
        help="Output directory for v3 graph files"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check for v2 database without modifying"
    )

    args = parser.parse_args()

    # Database migration
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    # Load sqlite-vec for the connection
    import sqlite_vec
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row

    is_v2 = detect_v2_db(conn)
    print(f"Database: {db_path}")
    print(f"Detected as v2: {is_v2}")

    if args.dry_run:
        if is_v2:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            protected = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE protected = 1"
            ).fetchone()[0]
            print(f"Total memories: {total}")
            print(f"Protected: {protected}")
            print("Run without --dry-run to migrate.")
        conn.close()
        return

    if is_v2:
        print("Migrating database v2 -> v3...")
        stats = migrate_v2_to_v3(conn)
        print(f"Migration complete:")
        print(f"  Columns added: {stats['columns_added']}")
        print(f"  Constitutional (L1): {stats['constitutional_count']}")
        print(f"  Legislative (L2): {stats['legislative_count']}")
        print(f"  Factual (L3): {stats['factual_count']}")
        print(f"  Ephemeral (L4): {stats['ephemeral_count']}")
    else:
        # Check if it's already v3
        try:
            version = conn.execute(
                "SELECT version FROM schema_version ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if version:
                print(f"Database is already v3 (version {version[0]}). Nothing to do.")
            else:
                print("Database schema not recognized. Manual inspection needed.")
        except Exception:
            print("Database schema not recognized. Manual inspection needed.")

    conn.close()

    # Graph migration
    if args.graph:
        graph_path = Path(args.graph)
        if not graph_path.exists():
            print(f"Graph file not found: {graph_path}")
            sys.exit(1)

        print(f"\nMigrating graph: {graph_path} -> {args.output_dir}")
        graph_stats = migrate_graph_v2_to_v3(graph_path, args.output_dir)

        if "error" in graph_stats:
            print(f"Error: {graph_stats['error']}")
            sys.exit(1)

        print(f"  v2 nodes: {graph_stats['v2_nodes']}")
        print(f"  v2 edges: {graph_stats['v2_edges']}")
        for name, info in graph_stats["graphs_created"].items():
            print(f"  {name}: {info['edges']} edges -> {info['path']}")
        if graph_stats["unmapped_edge_types"]:
            print(f"  Unmapped types (routed to semantic): "
                  f"{graph_stats['unmapped_edge_types']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
