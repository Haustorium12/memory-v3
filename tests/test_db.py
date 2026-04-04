"""
Tests for memory_v3.db -- the database layer.

Covers schema creation, CRUD, search, sensory buffer, write queue,
action logging, and v3-specific stats.
"""

import json

import pytest

from memory_v3 import db
from tests.conftest import _mock_embed


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestInitDb:
    """Schema creation and table existence."""

    EXPECTED_TABLES = [
        "memories",
        "memory_fts",
        "graveyard",
        "agent_offsets",
        "file_hashes",
        "conflicts",
        "compaction_receipts",
        # v3 new tables
        "sensory_buffer",
        "cluster_centroids",
        "action_log",
        "consolidation_log",
        "write_queue",
        "schema_version",
        # vec0 virtual table
        "memory_vec",
    ]

    def test_init_db(self, db_conn):
        """init_db should create all v2 + v3 tables."""
        rows = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        table_names = {row["name"] for row in rows}

        for expected in self.EXPECTED_TABLES:
            assert expected in table_names, f"Missing table: {expected}"

    def test_schema_version_recorded(self, db_conn):
        """A fresh init should record schema version 3."""
        row = db_conn.execute(
            "SELECT version FROM schema_version ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row["version"] == 3


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

class TestAddAndGetMemory:
    """Round-trip: add then get, including v3 fields."""

    def test_add_and_get_memory(self, db_conn, mock_embedding):
        emb = mock_embedding("test memory content")
        mid = db.add_memory(
            db_conn,
            content="test memory content",
            embedding=emb,
            content_type="fact",
            governance_layer=2,
            structure_type="atomic",
            surprise_score=0.8,
        )
        assert mid > 0

        mem = db.get_memory(db_conn, mid)
        assert mem is not None
        assert mem["content"] == "test memory content"
        assert mem["governance_layer"] == 2
        assert mem["structure_type"] == "atomic"
        assert abs(mem["surprise_score"] - 0.8) < 0.01

    def test_get_nonexistent_memory(self, db_conn):
        assert db.get_memory(db_conn, 99999) is None

    def test_add_protected_auto_governance(self, db_conn, mock_embedding):
        """Protected memories auto-classify to governance_layer 1."""
        emb = mock_embedding("identity core")
        mid = db.add_memory(
            db_conn,
            content="identity core",
            embedding=emb,
            protected=True,
        )
        mem = db.get_memory(db_conn, mid)
        assert mem["governance_layer"] == 1
        assert mem["protected"] == 1


class TestUpdateMemory:
    """update_memory with v3 fields."""

    def test_update_memory_governance(self, db_conn, mock_embedding):
        emb = mock_embedding("some fact")
        mid = db.add_memory(db_conn, content="some fact", embedding=emb)

        success = db.update_memory(db_conn, mid, governance_layer=1)
        assert success is True

        mem = db.get_memory(db_conn, mid)
        assert mem["governance_layer"] == 1

    def test_update_verification_status(self, db_conn, mock_embedding):
        emb = mock_embedding("verify me")
        mid = db.add_memory(db_conn, content="verify me", embedding=emb)

        db.update_memory(db_conn, mid, verification_status="verified")
        mem = db.get_memory(db_conn, mid)
        assert mem["verification_status"] == "verified"
        assert mem["last_verified_at"] is not None

    def test_update_no_fields_returns_false(self, db_conn, mock_embedding):
        emb = mock_embedding("noop")
        mid = db.add_memory(db_conn, content="noop", embedding=emb)
        assert db.update_memory(db_conn, mid) is False


# ---------------------------------------------------------------------------
# Archival
# ---------------------------------------------------------------------------

class TestArchiveMemory:
    """archive_memory respects governance and protected rules."""

    def test_archive_memory(self, db_conn, mock_embedding):
        """A normal memory (layer 3) can be archived."""
        emb = mock_embedding("ephemeral note")
        mid = db.add_memory(
            db_conn,
            content="ephemeral note",
            embedding=emb,
            governance_layer=4,
        )
        result = db.archive_memory(db_conn, mid, reason="test")
        assert result is True

        # Should not be retrievable
        assert db.get_memory(db_conn, mid) is None

        # Should be in graveyard
        grave = db_conn.execute(
            "SELECT * FROM graveyard WHERE memory_id = ?", (mid,)
        ).fetchone()
        assert grave is not None

    def test_archive_protected_blocked(self, db_conn, mock_embedding):
        """Protected memories should NOT be archived."""
        emb = mock_embedding("protected")
        mid = db.add_memory(
            db_conn,
            content="protected",
            embedding=emb,
            protected=True,
        )
        result = db.archive_memory(db_conn, mid, reason="test")
        assert result is False

    def test_archive_constitutional(self, db_conn, mock_embedding):
        """
        governance_layer=1 (constitutional) should block archival at the DB level.
        The archive function checks governance_layer in (1, 2) and refuses.
        """
        emb = mock_embedding("constitutional mem")
        mid = db.add_memory(
            db_conn,
            content="constitutional mem",
            embedding=emb,
            governance_layer=1,
        )
        # Force protected=0 so we test governance_layer check specifically
        db_conn.execute("UPDATE memories SET protected = 0 WHERE id = ?", (mid,))
        db_conn.commit()

        result = db.archive_memory(db_conn, mid, reason="test")
        assert result is False


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestHybridSearch:
    """hybrid_search with BM25 + vector fusion."""

    def test_hybrid_search(self, db_conn, mock_embedding):
        """Add 3 memories, search should return ranked results."""
        contents = [
            "Python is a programming language",
            "SQLite is an embedded database",
            "Python and SQLite work well together",
        ]
        for c in contents:
            db.add_memory(db_conn, content=c, embedding=mock_embedding(c))

        query = "Python programming"
        results = db.hybrid_search(
            db_conn,
            query_text=query,
            query_embedding=mock_embedding(query),
            limit=10,
        )
        assert len(results) > 0
        # Results should have hybrid_score
        assert "hybrid_score" in results[0]

    def test_hybrid_search_empty_db(self, db_conn, mock_embedding):
        results = db.hybrid_search(
            db_conn,
            query_text="anything",
            query_embedding=mock_embedding("anything"),
        )
        assert results == []


class TestKeywordSearch:
    """Pure BM25 keyword search via FTS5."""

    def test_keyword_search(self, db_conn, mock_embedding):
        db.add_memory(db_conn, content="Rust memory safety", embedding=mock_embedding("Rust memory safety"))
        db.add_memory(db_conn, content="Go concurrency model", embedding=mock_embedding("Go concurrency model"))

        results = db.keyword_search(db_conn, keywords="Rust")
        assert len(results) >= 1
        assert "Rust" in results[0]["content"]

    def test_keyword_search_no_match(self, db_conn, mock_embedding):
        db.add_memory(db_conn, content="hello world", embedding=mock_embedding("hello world"))
        results = db.keyword_search(db_conn, keywords="zzznonexistent")
        assert results == []


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestGetStats:
    """get_stats includes v3 governance_layer_counts."""

    def test_get_stats_v3(self, db_conn, mock_embedding):
        # Add memories in different governance layers
        db.add_memory(db_conn, content="identity fact", embedding=mock_embedding("identity fact"), governance_layer=1)
        db.add_memory(db_conn, content="a decision", embedding=mock_embedding("a decision"), governance_layer=2)
        db.add_memory(db_conn, content="plain fact", embedding=mock_embedding("plain fact"), governance_layer=3)
        db.add_memory(db_conn, content="an episode", embedding=mock_embedding("an episode"), governance_layer=4)

        stats = db.get_stats(db_conn)

        assert stats["total_active"] == 4
        assert "governance_layer_counts" in stats
        assert stats["governance_layer_counts"][1] == 1
        assert stats["governance_layer_counts"][2] == 1
        assert stats["governance_layer_counts"][3] == 1
        assert stats["governance_layer_counts"][4] == 1

    def test_stats_sensory_and_write_queue(self, db_conn, mock_embedding):
        """Stats should report pending sensory items and write queue items."""
        db.add_to_sensory_buffer(db_conn, content="buf item", content_hash="abc123")
        db.enqueue_write(db_conn, operation="add", payload={"content": "test"})

        stats = db.get_stats(db_conn)
        assert stats["sensory_buffer_pending"] == 1
        assert stats["write_queue_pending"] == 1


# ---------------------------------------------------------------------------
# Sensory Buffer
# ---------------------------------------------------------------------------

class TestSensoryBuffer:
    """add_to_sensory_buffer, get_pending_sensory, mark_sensory_processed."""

    def test_add_to_sensory_buffer(self, db_conn):
        buf_id = db.add_to_sensory_buffer(
            db_conn,
            content="new observation",
            content_hash="hash123",
            content_type="episode",
            source="conversation",
            novelty_score=0.9,
        )
        assert buf_id > 0

    def test_get_pending_sensory(self, db_conn):
        db.add_to_sensory_buffer(db_conn, content="item 1", content_hash="h1")
        db.add_to_sensory_buffer(db_conn, content="item 2", content_hash="h2")

        pending = db.get_pending_sensory(db_conn)
        assert len(pending) == 2
        assert pending[0]["content"] == "item 1"

    def test_mark_sensory_processed(self, db_conn):
        buf_id = db.add_to_sensory_buffer(db_conn, content="process me", content_hash="hp")

        db.mark_sensory_processed(
            db_conn,
            buffer_id=buf_id,
            processed=1,
            decision="ADD",
            decision_reason="novel content",
        )

        # Should no longer appear in pending
        pending = db.get_pending_sensory(db_conn)
        assert len(pending) == 0

        # Verify the record was updated
        row = db_conn.execute(
            "SELECT * FROM sensory_buffer WHERE id = ?", (buf_id,)
        ).fetchone()
        assert row["processed"] == 1
        assert row["decision"] == "ADD"


# ---------------------------------------------------------------------------
# Write Queue
# ---------------------------------------------------------------------------

class TestWriteQueue:
    """enqueue_write, get_pending_writes, mark_write_complete."""

    def test_enqueue_write(self, db_conn):
        wid = db.enqueue_write(
            db_conn,
            operation="add_memory",
            payload={"content": "queued memory"},
            priority=3,
        )
        assert wid > 0

    def test_get_pending_writes(self, db_conn):
        db.enqueue_write(db_conn, operation="add", payload={"a": 1}, priority=5)
        db.enqueue_write(db_conn, operation="update", payload={"b": 2}, priority=1)

        pending = db.get_pending_writes(db_conn)
        assert len(pending) == 2
        # Lower priority number = higher priority = comes first
        assert pending[0]["priority"] == 1

    def test_mark_write_complete(self, db_conn):
        wid = db.enqueue_write(db_conn, operation="add", payload="test")

        db.mark_write_complete(db_conn, wid, status="completed")

        pending = db.get_pending_writes(db_conn)
        assert len(pending) == 0

        row = db_conn.execute(
            "SELECT * FROM write_queue WHERE id = ?", (wid,)
        ).fetchone()
        assert row["status"] == "completed"
        assert row["completed_at"] is not None

    def test_mark_write_failed(self, db_conn):
        wid = db.enqueue_write(db_conn, operation="add", payload="fail test")

        db.mark_write_complete(db_conn, wid, status="failed", error="disk full")

        row = db_conn.execute(
            "SELECT * FROM write_queue WHERE id = ?", (wid,)
        ).fetchone()
        assert row["status"] == "failed"
        assert row["error"] == "disk full"


# ---------------------------------------------------------------------------
# Action Log
# ---------------------------------------------------------------------------

class TestActionLog:
    """log_action round-trip."""

    def test_log_action(self, db_conn):
        log_id = db.log_action(
            db_conn,
            new_fact="Sean prefers Python",
            similar_ids=[1, 2, 3],
            action="ADD",
            confidence=0.95,
            model_version="v3-test",
        )
        assert log_id > 0

        row = db_conn.execute(
            "SELECT * FROM action_log WHERE id = ?", (log_id,)
        ).fetchone()
        assert row["new_fact"] == "Sean prefers Python"
        assert row["action"] == "ADD"
        assert json.loads(row["similar_ids"]) == [1, 2, 3]
        assert abs(row["confidence"] - 0.95) < 0.01
        assert row["model_version"] == "v3-test"
