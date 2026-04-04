"""
memory-v3 -- Database Layer (sqlite-vec + FTS5)
Extended from v2: governance layers, sensory buffer, write queue,
action logging, consolidation tracking, and verification status.
"""

import sqlite3
import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec
import numpy as np

from .config import get_config

# ---------- constants ----------

EMBEDDING_DIM = int(os.environ.get("MEMORY_V3_EMBED_DIM", "768"))


def _serialize_f32(vec: list[float]) -> bytes:
    """Serialize a list of floats to raw bytes for sqlite-vec."""
    return np.array(vec, dtype=np.float32).tobytes()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- connection factory ----------

def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Get a SQLite connection with sqlite-vec loaded."""
    if db_path is None:
        db_path = get_config().db_path
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------- schema ----------

SCHEMA_SQL = """
-- Core memories table (v2 columns + v3 extensions)
CREATE TABLE IF NOT EXISTS memories (
    -- v2 core columns
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    content_type TEXT DEFAULT 'fact',
    source_file TEXT,
    source_line INTEGER,
    author TEXT DEFAULT 'claude-code',
    authority_level INTEGER DEFAULT 2,
    confidence REAL DEFAULT 0.95,
    protected INTEGER DEFAULT 0,
    tags TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    access_count INTEGER DEFAULT 1,
    activation_score REAL DEFAULT 0.0,
    importance_score REAL DEFAULT 0.5,
    decay_rate REAL DEFAULT 0.1,
    archived INTEGER DEFAULT 0,
    supersedes INTEGER,
    content_hash TEXT,

    -- v3 new columns
    governance_layer INTEGER DEFAULT 3,
    confidence_floor REAL DEFAULT 0.0,
    last_verified_at TEXT,
    verification_status TEXT DEFAULT 'unverified',
    cross_ref_count INTEGER DEFAULT 0,
    structure_type TEXT DEFAULT 'narrative',
    surprise_score REAL DEFAULT 0.5,
    memory_layer TEXT DEFAULT 'STM',
    cluster_id INTEGER,
    linked_memories TEXT DEFAULT '[]',
    link_descriptions TEXT DEFAULT '{}',
    source_hash TEXT
);

-- FTS5 full-text search index
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    tags,
    source_file,
    content='memories',
    content_rowid='id'
);

-- Triggers to keep FTS5 in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(rowid, content, tags, source_file)
    VALUES (new.id, new.content, new.tags, new.source_file);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, tags, source_file)
    VALUES ('delete', old.id, old.content, old.tags, old.source_file);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, tags, source_file)
    VALUES ('delete', old.id, old.content, old.tags, old.source_file);
    INSERT INTO memory_fts(rowid, content, tags, source_file)
    VALUES (new.id, new.content, new.tags, new.source_file);
END;

-- Graveyard for archived memories
CREATE TABLE IF NOT EXISTS graveyard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL,
    archived_at TEXT NOT NULL,
    reason TEXT,
    last_activation_score REAL
);

-- Consumer offsets for multi-agent sync
CREATE TABLE IF NOT EXISTS agent_offsets (
    agent_id TEXT PRIMARY KEY,
    last_read_line INTEGER DEFAULT 0,
    last_read_time TEXT
);

-- File hash registry (for incremental indexing)
CREATE TABLE IF NOT EXISTS file_hashes (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0
);

-- Conflict log
CREATE TABLE IF NOT EXISTS conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id_a INTEGER,
    memory_id_b INTEGER,
    agent_a TEXT,
    agent_b TEXT,
    description TEXT,
    resolved INTEGER DEFAULT 0,
    resolved_by TEXT,
    created_at TEXT NOT NULL
);

-- Compaction receipts
CREATE TABLE IF NOT EXISTS compaction_receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    original_tokens INTEGER,
    compressed_tokens INTEGER,
    ratio REAL,
    protected_items_extracted INTEGER,
    verification_score REAL,
    vault_files_updated TEXT,
    receipt_data TEXT
);

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

-- v2 Indexes
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(content_type);
CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);
CREATE INDEX IF NOT EXISTS idx_memories_protected ON memories(protected);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source_file);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_activation ON memories(activation_score);

-- v3 Indexes
CREATE INDEX IF NOT EXISTS idx_memories_governance ON memories(governance_layer);
CREATE INDEX IF NOT EXISTS idx_memories_memory_layer ON memories(memory_layer);
CREATE INDEX IF NOT EXISTS idx_memories_cluster ON memories(cluster_id);
CREATE INDEX IF NOT EXISTS idx_memories_verification ON memories(verification_status);
CREATE INDEX IF NOT EXISTS idx_sensory_processed ON sensory_buffer(processed);
CREATE INDEX IF NOT EXISTS idx_write_queue_status ON write_queue(status, priority);
CREATE INDEX IF NOT EXISTS idx_action_log_action ON action_log(action);
"""


def init_vec_table(conn: sqlite3.Connection):
    """Create the vec0 virtual table for vector search."""
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}]
        )
    """)
    conn.commit()


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Initialize the database with full v3 schema."""
    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    init_vec_table(conn)

    # Record schema version if not already present
    existing = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    if existing == 0:
        conn.execute(
            "INSERT INTO schema_version (version, migrated_at, notes) VALUES (?, ?, ?)",
            (3, _now_iso(), "Fresh v3 init")
        )
    conn.commit()
    return conn


# ---------- CRUD operations ----------

def add_memory(
    conn: sqlite3.Connection,
    content: str,
    embedding: list[float],
    content_type: str = "fact",
    source_file: str | None = None,
    source_line: int | None = None,
    author: str = "claude-code",
    authority_level: int = 2,
    confidence: float = 0.95,
    protected: bool = False,
    tags: list[str] | None = None,
    importance_score: float = 0.5,
    decay_rate: float = 0.1,
    supersedes: int | None = None,
    # v3 new params
    governance_layer: int = 3,
    structure_type: str = "narrative",
    surprise_score: float = 0.5,
    source_hash: str | None = None,
) -> int:
    """Add a new memory with embedding. Returns the memory ID."""
    now = _now_iso()
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    tags_json = json.dumps(tags or [])

    # Auto-determine governance layer from protected/type if not explicitly set
    if protected and governance_layer == 3:
        governance_layer = 1  # Protected memories are constitutional

    cursor = conn.execute(
        """INSERT INTO memories
        (content, content_type, source_file, source_line, author,
         authority_level, confidence, protected, tags,
         created_at, updated_at, last_accessed_at,
         importance_score, decay_rate, supersedes, content_hash,
         governance_layer, structure_type, surprise_score, source_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (content, content_type, source_file, source_line, author,
         authority_level, confidence, 1 if protected else 0, tags_json,
         now, now, now,
         importance_score, decay_rate, supersedes, content_hash,
         governance_layer, structure_type, surprise_score, source_hash)
    )
    memory_id = cursor.lastrowid

    # Insert embedding into vec0
    conn.execute(
        "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
        (memory_id, _serialize_f32(embedding))
    )
    conn.commit()
    return memory_id


def get_memory(conn: sqlite3.Connection, memory_id: int) -> dict | None:
    """Retrieve a single memory by ID. Updates access timestamp."""
    row = conn.execute(
        "SELECT * FROM memories WHERE id = ? AND archived = 0",
        (memory_id,)
    ).fetchone()
    if row is None:
        return None

    # Update access stats
    now = _now_iso()
    conn.execute(
        """UPDATE memories
        SET last_accessed_at = ?, access_count = access_count + 1
        WHERE id = ?""",
        (now, memory_id)
    )
    conn.commit()
    return dict(row)


def update_memory(
    conn: sqlite3.Connection,
    memory_id: int,
    content: str | None = None,
    tags: list[str] | None = None,
    protected: bool | None = None,
    importance_score: float | None = None,
    # v3 new params
    governance_layer: int | None = None,
    structure_type: str | None = None,
    verification_status: str | None = None,
    confidence: float | None = None,
    memory_layer: str | None = None,
    cluster_id: int | None = None,
    linked_memories: list[int] | None = None,
    link_descriptions: dict | None = None,
    cross_ref_count: int | None = None,
    surprise_score: float | None = None,
) -> bool:
    """Update an existing memory. Only provided fields are changed."""
    updates = []
    params = []

    if content is not None:
        updates.append("content = ?")
        params.append(content)
        updates.append("content_hash = ?")
        params.append(hashlib.sha256(content.encode()).hexdigest()[:16])
    if tags is not None:
        updates.append("tags = ?")
        params.append(json.dumps(tags))
    if protected is not None:
        updates.append("protected = ?")
        params.append(1 if protected else 0)
    if importance_score is not None:
        updates.append("importance_score = ?")
        params.append(importance_score)
    if governance_layer is not None:
        updates.append("governance_layer = ?")
        params.append(governance_layer)
    if structure_type is not None:
        updates.append("structure_type = ?")
        params.append(structure_type)
    if verification_status is not None:
        updates.append("verification_status = ?")
        params.append(verification_status)
        if verification_status == "verified":
            updates.append("last_verified_at = ?")
            params.append(_now_iso())
    if confidence is not None:
        updates.append("confidence = ?")
        params.append(confidence)
    if memory_layer is not None:
        updates.append("memory_layer = ?")
        params.append(memory_layer)
    if cluster_id is not None:
        updates.append("cluster_id = ?")
        params.append(cluster_id)
    if linked_memories is not None:
        updates.append("linked_memories = ?")
        params.append(json.dumps(linked_memories))
    if link_descriptions is not None:
        updates.append("link_descriptions = ?")
        params.append(json.dumps(link_descriptions))
    if cross_ref_count is not None:
        updates.append("cross_ref_count = ?")
        params.append(cross_ref_count)
    if surprise_score is not None:
        updates.append("surprise_score = ?")
        params.append(surprise_score)

    if not updates:
        return False

    updates.append("updated_at = ?")
    params.append(_now_iso())
    params.append(memory_id)

    conn.execute(
        f"UPDATE memories SET {', '.join(updates)} WHERE id = ?",
        params
    )
    conn.commit()
    return True


def archive_memory(
    conn: sqlite3.Connection,
    memory_id: int,
    reason: str = "decay"
) -> bool:
    """Archive a memory to the graveyard. Respects governance layer rules."""
    row = conn.execute(
        "SELECT * FROM memories WHERE id = ? AND archived = 0",
        (memory_id,)
    ).fetchone()
    if row is None:
        return False

    mem = dict(row)

    # Never archive protected memories
    if mem["protected"]:
        return False

    # Never archive constitutional or legislative layers
    if mem.get("governance_layer", 3) in (1, 2):
        return False

    # Write to graveyard
    conn.execute(
        """INSERT INTO graveyard
        (memory_id, content, metadata, archived_at, reason, last_activation_score)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (memory_id, mem["content"], json.dumps(mem), _now_iso(),
         reason, mem["activation_score"])
    )

    # Mark as archived
    conn.execute("UPDATE memories SET archived = 1 WHERE id = ?", (memory_id,))

    # Remove from vec index
    conn.execute("DELETE FROM memory_vec WHERE id = ?", (memory_id,))
    conn.commit()
    return True


# ---------- Search ----------

def hybrid_search(
    conn: sqlite3.Connection,
    query_text: str,
    query_embedding: list[float],
    limit: int = 10,
    content_type: str | None = None,
    include_archived: bool = False,
    governance_layer: int | None = None,
) -> list[dict]:
    """
    Hybrid search: BM25 (FTS5) + vector similarity (vec0).
    Uses Reciprocal Rank Fusion to combine results.
    """
    cfg = get_config()
    k = cfg.rrf_k

    # --- BM25 keyword search ---
    fts_results = {}
    try:
        rows = conn.execute(
            """SELECT rowid, rank FROM memory_fts
            WHERE memory_fts MATCH ?
            ORDER BY rank LIMIT ?""",
            (query_text, limit * 3)
        ).fetchall()
        for rank_pos, row in enumerate(rows):
            fts_results[row["rowid"]] = 1.0 / (k + rank_pos + 1)
    except Exception:
        pass  # FTS match syntax errors are non-fatal

    # --- Vector similarity search ---
    vec_results = {}
    rows = conn.execute(
        """SELECT id, distance FROM memory_vec
        WHERE embedding MATCH ?
        ORDER BY distance LIMIT ?""",
        (_serialize_f32(query_embedding), limit * 3)
    ).fetchall()
    for rank_pos, row in enumerate(rows):
        vec_results[row["id"]] = 1.0 / (k + rank_pos + 1)

    # --- Reciprocal Rank Fusion ---
    all_ids = set(fts_results.keys()) | set(vec_results.keys())
    fused = []
    for mid in all_ids:
        score = fts_results.get(mid, 0) + vec_results.get(mid, 0)
        fused.append((mid, score))

    fused.sort(key=lambda x: x[1], reverse=True)

    # Build filter clauses
    filters = []
    if not include_archived:
        filters.append("archived = 0")
    if content_type:
        filters.append(f"content_type = '{content_type}'")
    if governance_layer is not None:
        filters.append(f"governance_layer = {governance_layer}")

    where_clause = " AND ".join(filters) if filters else "1=1"

    # Fetch full records
    results = []
    for mid, score in fused[:limit * 2]:
        row = conn.execute(
            f"SELECT * FROM memories WHERE id = ? AND {where_clause}",
            (mid,)
        ).fetchone()
        if row:
            mem = dict(row)
            mem["hybrid_score"] = score
            mem["bm25_score"] = fts_results.get(mid, 0)
            mem["vec_score"] = vec_results.get(mid, 0)
            results.append(mem)

        if len(results) >= limit:
            break

    # Update access timestamps for returned results
    now = _now_iso()
    for mem in results:
        conn.execute(
            """UPDATE memories
            SET last_accessed_at = ?, access_count = access_count + 1
            WHERE id = ?""",
            (now, mem["id"])
        )
    conn.commit()

    return results


def keyword_search(
    conn: sqlite3.Connection,
    keywords: str,
    limit: int = 10,
) -> list[dict]:
    """Pure BM25 keyword search via FTS5."""
    try:
        rows = conn.execute(
            """SELECT rowid, rank FROM memory_fts
            WHERE memory_fts MATCH ?
            ORDER BY rank LIMIT ?""",
            (keywords, limit)
        ).fetchall()
    except Exception:
        return []

    results = []
    now = _now_iso()
    for row in rows:
        mem_row = conn.execute(
            "SELECT * FROM memories WHERE id = ? AND archived = 0",
            (row["rowid"],)
        ).fetchone()
        if mem_row:
            mem = dict(mem_row)
            mem["bm25_rank"] = row["rank"]
            results.append(mem)
            conn.execute(
                """UPDATE memories
                SET last_accessed_at = ?, access_count = access_count + 1
                WHERE id = ?""",
                (now, mem["id"])
            )
    conn.commit()
    return results


def list_recent(
    conn: sqlite3.Connection,
    hours: int = 24,
    limit: int = 20,
) -> list[dict]:
    """Get recently created or accessed memories."""
    rows = conn.execute(
        """SELECT * FROM memories
        WHERE archived = 0
        ORDER BY created_at DESC
        LIMIT ?""",
        (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_stats(conn: sqlite3.Connection) -> dict:
    """Get database statistics including v3 governance layer counts."""
    total = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE archived = 0"
    ).fetchone()[0]
    archived = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE archived = 1"
    ).fetchone()[0]
    protected = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE protected = 1 AND archived = 0"
    ).fetchone()[0]
    graveyard_count = conn.execute(
        "SELECT COUNT(*) FROM graveyard"
    ).fetchone()[0]
    files_indexed = conn.execute(
        "SELECT COUNT(*) FROM file_hashes"
    ).fetchone()[0]

    # Type counts
    type_counts = {}
    for row in conn.execute(
        "SELECT content_type, COUNT(*) as cnt FROM memories "
        "WHERE archived = 0 GROUP BY content_type"
    ).fetchall():
        type_counts[row["content_type"]] = row["cnt"]

    # v3: Governance layer counts
    layer_counts = {}
    for row in conn.execute(
        "SELECT governance_layer, COUNT(*) as cnt FROM memories "
        "WHERE archived = 0 GROUP BY governance_layer"
    ).fetchall():
        layer_counts[row["governance_layer"]] = row["cnt"]

    # v3: Memory layer counts (STM/LTM)
    memory_layer_counts = {}
    for row in conn.execute(
        "SELECT memory_layer, COUNT(*) as cnt FROM memories "
        "WHERE archived = 0 GROUP BY memory_layer"
    ).fetchall():
        memory_layer_counts[row["memory_layer"]] = row["cnt"]

    # v3: Verification status counts
    verification_counts = {}
    for row in conn.execute(
        "SELECT verification_status, COUNT(*) as cnt FROM memories "
        "WHERE archived = 0 GROUP BY verification_status"
    ).fetchall():
        verification_counts[row["verification_status"]] = row["cnt"]

    # v3: Sensory buffer stats
    sensory_pending = conn.execute(
        "SELECT COUNT(*) FROM sensory_buffer WHERE processed = 0"
    ).fetchone()[0]

    # v3: Write queue stats
    write_queue_pending = conn.execute(
        "SELECT COUNT(*) FROM write_queue WHERE status = 'pending'"
    ).fetchone()[0]

    # v3: Cluster count
    cluster_count = conn.execute(
        "SELECT COUNT(DISTINCT cluster_id) FROM memories "
        "WHERE archived = 0 AND cluster_id IS NOT NULL"
    ).fetchone()[0]

    return {
        "total_active": total,
        "archived": archived,
        "protected": protected,
        "graveyard": graveyard_count,
        "files_indexed": files_indexed,
        "type_counts": type_counts,
        "governance_layer_counts": layer_counts,
        "memory_layer_counts": memory_layer_counts,
        "verification_counts": verification_counts,
        "sensory_buffer_pending": sensory_pending,
        "write_queue_pending": write_queue_pending,
        "cluster_count": cluster_count,
    }


# ---------- Sensory Buffer ----------

def add_to_sensory_buffer(
    conn: sqlite3.Connection,
    content: str,
    content_hash: str,
    content_type: str = "fact",
    source: str | None = None,
    author: str = "claude-code",
    topic_cluster: str | None = None,
    novelty_score: float = 0.5,
) -> int:
    """Add an item to the sensory buffer for pre-encoding evaluation.
    Returns the buffer item ID."""
    cursor = conn.execute(
        """INSERT INTO sensory_buffer
        (content, content_hash, content_type, source, author,
         topic_cluster, novelty_score, received_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (content, content_hash, content_type, source, author,
         topic_cluster, novelty_score, _now_iso())
    )
    conn.commit()
    return cursor.lastrowid


def get_pending_sensory(
    conn: sqlite3.Connection,
    limit: int = 50,
) -> list[dict]:
    """Get unprocessed items from the sensory buffer."""
    rows = conn.execute(
        """SELECT * FROM sensory_buffer
        WHERE processed = 0
        ORDER BY received_at ASC
        LIMIT ?""",
        (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def mark_sensory_processed(
    conn: sqlite3.Connection,
    buffer_id: int,
    processed: int = 1,
    decision: str | None = None,
    decision_reason: str | None = None,
):
    """Mark a sensory buffer item as processed with optional decision info."""
    conn.execute(
        """UPDATE sensory_buffer
        SET processed = ?, processed_at = ?, decision = ?, decision_reason = ?
        WHERE id = ?""",
        (processed, _now_iso(), decision, decision_reason, buffer_id)
    )
    conn.commit()


# ---------- Write Queue ----------

def enqueue_write(
    conn: sqlite3.Connection,
    operation: str,
    payload: dict | str,
    priority: int = 5,
) -> int:
    """Enqueue an async write operation. Returns the write queue ID."""
    if isinstance(payload, dict):
        payload = json.dumps(payload)
    cursor = conn.execute(
        """INSERT INTO write_queue (operation, payload, priority, created_at)
        VALUES (?, ?, ?, ?)""",
        (operation, payload, priority, _now_iso())
    )
    conn.commit()
    return cursor.lastrowid


def get_pending_writes(
    conn: sqlite3.Connection,
    limit: int = 10,
) -> list[dict]:
    """Get pending write operations ordered by priority (lower = higher priority)."""
    rows = conn.execute(
        """SELECT * FROM write_queue
        WHERE status = 'pending'
        ORDER BY priority ASC, created_at ASC
        LIMIT ?""",
        (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def mark_write_complete(
    conn: sqlite3.Connection,
    write_id: int,
    status: str = "completed",
    error: str | None = None,
):
    """Mark a write queue item as completed or failed."""
    conn.execute(
        """UPDATE write_queue
        SET status = ?, error = ?, completed_at = ?
        WHERE id = ?""",
        (status, error, _now_iso(), write_id)
    )
    conn.commit()


# ---------- Action Log ----------

def log_action(
    conn: sqlite3.Connection,
    new_fact: str,
    similar_ids: list[int],
    action: str,
    confidence: float = 0.0,
    model_version: str | None = None,
) -> int:
    """Log an extraction decision (ADD/UPDATE/DELETE/NONE). Returns log entry ID."""
    cursor = conn.execute(
        """INSERT INTO action_log
        (timestamp, new_fact, similar_ids, action, confidence, model_version)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (_now_iso(), new_fact, json.dumps(similar_ids), action,
         confidence, model_version)
    )
    conn.commit()
    return cursor.lastrowid


# ---------- Consolidation Log ----------

def log_consolidation(
    conn: sqlite3.Connection,
    cycle_type: str,
    stats: dict,
) -> int:
    """Log a consolidation cycle (merge/split/prune). Returns log entry ID."""
    cursor = conn.execute(
        """INSERT INTO consolidation_log (timestamp, cycle_type, stats)
        VALUES (?, ?, ?)""",
        (_now_iso(), cycle_type, json.dumps(stats))
    )
    conn.commit()
    return cursor.lastrowid
