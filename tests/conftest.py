"""
memory-v3 test fixtures.

All tests run WITHOUT Ollama -- embeddings are deterministic mocks
derived from SHA-256 hashes of the input text, normalized to unit length.
"""

import hashlib
import os
import tempfile

import numpy as np
import pytest
import sqlite3
import sqlite_vec


# ---------------------------------------------------------------------------
# Mock embedding function (deterministic, 768-dim from SHA-256)
# ---------------------------------------------------------------------------

def _mock_embed(text: str) -> list[float]:
    """Generate a deterministic 768-dim embedding from a SHA-256 hash."""
    h = hashlib.sha256(text.encode()).digest()
    # Tile the 32-byte hash to fill 768 floats
    arr = np.frombuffer(h * 96, dtype=np.uint8)[:768].astype(np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedding():
    """Return the mock embedding function for direct use in tests."""
    return _mock_embed


@pytest.fixture
def db_conn():
    """
    In-memory-style SQLite connection with the FULL v3 schema.

    Uses a temp file because sqlite-vec's vec0 virtual table may not
    work reliably with :memory: databases on all platforms.
    """
    from memory_v3 import db

    tmp = tempfile.mktemp(suffix=".db")
    conn = db.init_db(tmp)
    yield conn
    conn.close()
    try:
        os.unlink(tmp)
    except OSError:
        pass


@pytest.fixture(autouse=True)
def _patch_embeddings(monkeypatch):
    """
    Globally replace Ollama-based embedding functions with deterministic mocks.

    This is autouse so every test automatically gets mock embeddings --
    no Ollama server required.
    """
    import memory_v3.embeddings as emb

    monkeypatch.setattr(emb, "embed_text", _mock_embed)
    monkeypatch.setattr(emb, "embed_batch", lambda texts: [_mock_embed(t) for t in texts])
