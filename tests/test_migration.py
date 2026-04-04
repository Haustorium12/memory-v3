"""
Tests for memory_v3.migration -- v2 -> v3 database migration.

Covers detection, column addition, governance classification, table creation,
and idempotent re-running.
"""

import os
import sqlite3
import tempfile

import pytest
import sqlite_vec

from memory_v3.migration import (
    detect_v2_db,
    migrate_v2_to_v3,
    V3_NEW_COLUMNS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

V2_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
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
    content_hash TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content, tags, source_file,
    content='memories', content_rowid='id'
);

CREATE TABLE IF NOT EXISTS graveyard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL,
    archived_at TEXT NOT NULL,
    reason TEXT,
    last_activation_score REAL
);

CREATE TABLE IF NOT EXISTS agent_offsets (
    agent_id TEXT PRIMARY KEY,
    last_read_line INTEGER DEFAULT 0,
    last_read_time TEXT
);

CREATE TABLE IF NOT EXISTS file_hashes (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0
);

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
"""


def _make_v2_db() -> tuple[sqlite3.Connection, str]:
    """Create a temporary v2-schema database and return (conn, path)."""
    tmp = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(tmp, timeout=10)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    conn.executescript(V2_SCHEMA_SQL)
    conn.commit()
    return conn, tmp


def _insert_v2_memory(conn, content, content_type="fact", protected=0):
    """Insert a minimal v2 memory row."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO memories
        (content, content_type, protected, created_at, updated_at, last_accessed_at)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (content, content_type, protected, now, now, now),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestDetectV2Db:

    def test_detect_v2_db(self):
        """A v2 database (memories table without governance_layer) should be detected."""
        conn, tmp = _make_v2_db()
        try:
            assert detect_v2_db(conn) is True
        finally:
            conn.close()
            os.unlink(tmp)

    def test_detect_v3_db_is_not_v2(self, db_conn):
        """A v3 database (has governance_layer) should NOT be detected as v2."""
        assert detect_v2_db(db_conn) is False

    def test_detect_empty_db(self):
        """An empty database with no tables should NOT be detected as v2."""
        tmp = tempfile.mktemp(suffix=".db")
        conn = sqlite3.connect(tmp)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.row_factory = sqlite3.Row
        try:
            assert detect_v2_db(conn) is False
        finally:
            conn.close()
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Column Migration
# ---------------------------------------------------------------------------

class TestMigrateAddsColumns:

    def test_migrate_adds_columns(self):
        """Migration should add all 12 new v3 columns to the memories table."""
        conn, tmp = _make_v2_db()
        try:
            migrate_v2_to_v3(conn)

            cols = conn.execute("PRAGMA table_info(memories)").fetchall()
            col_names = {row[1] for row in cols}

            for col_name, _ in V3_NEW_COLUMNS:
                assert col_name in col_names, f"Missing column after migration: {col_name}"
        finally:
            conn.close()
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Governance Classification
# ---------------------------------------------------------------------------

class TestMigrateGovernanceClassification:

    def test_migrate_governance_classification(self):
        """Migration should classify protected->L1, decision->L2, episode->L4."""
        conn, tmp = _make_v2_db()
        try:
            _insert_v2_memory(conn, "I am Sean", content_type="identity", protected=1)
            _insert_v2_memory(conn, "We decided to use Python", content_type="decision", protected=0)
            _insert_v2_memory(conn, "Just a fact", content_type="fact", protected=0)
            _insert_v2_memory(conn, "A conversation happened", content_type="episode", protected=0)

            stats = migrate_v2_to_v3(conn)

            # Check governance layers were assigned
            assert stats["constitutional_count"] >= 1  # protected=1 -> L1
            assert stats["legislative_count"] >= 1      # decision -> L2
            assert stats["ephemeral_count"] >= 1        # episode -> L4

            # Verify actual rows
            row_identity = conn.execute(
                "SELECT governance_layer FROM memories WHERE content = 'I am Sean'"
            ).fetchone()
            assert row_identity[0] == 1

            row_decision = conn.execute(
                "SELECT governance_layer FROM memories WHERE content = 'We decided to use Python'"
            ).fetchone()
            assert row_decision[0] == 2

            row_episode = conn.execute(
                "SELECT governance_layer FROM memories WHERE content = 'A conversation happened'"
            ).fetchone()
            assert row_episode[0] == 4

            row_fact = conn.execute(
                "SELECT governance_layer FROM memories WHERE content = 'Just a fact'"
            ).fetchone()
            assert row_fact[0] == 3  # default
        finally:
            conn.close()
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# New Table Creation
# ---------------------------------------------------------------------------

class TestMigrateCreatesNewTables:

    V3_NEW_TABLES = [
        "sensory_buffer",
        "cluster_centroids",
        "action_log",
        "consolidation_log",
        "write_queue",
        "schema_version",
    ]

    def test_migrate_creates_new_tables(self):
        """Migration should create all 6 new v3 tables."""
        conn, tmp = _make_v2_db()
        try:
            migrate_v2_to_v3(conn)

            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
            table_names = {row[0] for row in rows}

            for table in self.V3_NEW_TABLES:
                assert table in table_names, f"Missing table after migration: {table}"
        finally:
            conn.close()
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Idempotent Migration
# ---------------------------------------------------------------------------

class TestMigrateIdempotent:

    def test_migrate_idempotent(self):
        """Running migration twice should not error."""
        conn, tmp = _make_v2_db()
        try:
            _insert_v2_memory(conn, "test memory", protected=0)

            stats1 = migrate_v2_to_v3(conn)
            assert stats1["columns_added"] == 12

            # Second run: columns already exist, should add 0
            stats2 = migrate_v2_to_v3(conn)
            assert stats2["columns_added"] == 0
        finally:
            conn.close()
            os.unlink(tmp)
