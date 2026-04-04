"""
memory-v3 -- Priority-based Async Write Queue

Background worker threads process writes from the write_queue table in
priority order (1=constitutional ... 4=ephemeral).  Each write is a JSON
payload describing the operation (add_memory, update_memory, archive_memory).

The WriteQueue class manages the worker pool.  Module-level enqueue() and
flush() functions provide fire-and-forget convenience without managing a
pool instance.
"""

import json
import logging
import threading
import time
from typing import Any

from .. import db, embeddings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WriteQueue class -- managed worker pool
# ---------------------------------------------------------------------------

class WriteQueue:
    """Priority-based async write queue backed by the write_queue table.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection (should be thread-safe with WAL mode).
    num_workers : int
        Number of background worker threads.
    batch_size : int
        Max writes fetched per worker iteration.
    """

    def __init__(self, conn, num_workers: int = 2, batch_size: int = 10):
        self.conn = conn
        self.num_workers = num_workers
        self.batch_size = batch_size
        self._running = False
        self._workers: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._stats = {"processed": 0, "failed": 0}

    # -- lifecycle -----------------------------------------------------------

    def start(self):
        """Start background worker threads."""
        if self._running:
            log.warning("WriteQueue already running")
            return
        self._running = True
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"write-worker-{i}",
            )
            t.start()
            self._workers.append(t)
        log.info("WriteQueue started with %d workers", self.num_workers)

    def stop(self, timeout: float = 5.0):
        """Stop background workers gracefully.

        Sets the running flag to False and waits up to *timeout* seconds
        for each worker to finish its current iteration.
        """
        self._running = False
        for t in self._workers:
            t.join(timeout=timeout)
        self._workers.clear()
        log.info("WriteQueue stopped")

    @property
    def running(self) -> bool:
        return self._running

    # -- enqueue / flush -----------------------------------------------------

    def enqueue(self, operation: str, payload: dict, priority: int = 3) -> int:
        """Add a write operation to the queue.

        Priority levels mirror governance layers:
            1 = constitutional, 2 = legislative, 3 = factual, 4 = ephemeral

        Returns the write_queue row id.
        """
        return db.enqueue_write(
            self.conn, operation, json.dumps(payload), priority
        )

    def flush(self) -> dict[str, int]:
        """Force-process all pending writes immediately (blocking).

        Returns a dict with ``processed`` and ``failed`` counts.
        """
        processed = 0
        failed = 0
        while True:
            pending = db.get_pending_writes(self.conn, limit=self.batch_size)
            if not pending:
                break
            for write in pending:
                try:
                    self._process_write(write)
                    db.mark_write_complete(
                        self.conn, write["id"], status="completed"
                    )
                    processed += 1
                except Exception as exc:
                    db.mark_write_complete(
                        self.conn, write["id"],
                        status="failed", error=str(exc),
                    )
                    failed += 1
                    log.error("Write %d failed: %s", write["id"], exc)
        return {"processed": processed, "failed": failed}

    # -- worker loop ---------------------------------------------------------

    def _worker_loop(self):
        """Background worker that processes writes from the queue."""
        while self._running:
            try:
                pending = db.get_pending_writes(
                    self.conn, limit=self.batch_size
                )
            except Exception as exc:
                log.error("Worker failed to fetch writes: %s", exc)
                time.sleep(1.0)
                continue

            if not pending:
                time.sleep(0.5)
                continue

            for write in pending:
                if not self._running:
                    break
                try:
                    self._process_write(write)
                    db.mark_write_complete(
                        self.conn, write["id"], status="completed"
                    )
                    with self._lock:
                        self._stats["processed"] += 1
                except Exception as exc:
                    db.mark_write_complete(
                        self.conn, write["id"],
                        status="failed", error=str(exc),
                    )
                    with self._lock:
                        self._stats["failed"] += 1
                    log.error("Write %d failed: %s", write["id"], exc)

    # -- operation dispatch --------------------------------------------------

    def _process_write(self, write: dict):
        """Execute a single write operation.

        Supported operations:
            add_memory     -- generate embedding, insert into memories + vec
            update_memory  -- update fields on an existing memory
            archive_memory -- move memory to graveyard
        """
        payload = json.loads(write["payload"])
        op = write["operation"]

        if op == "add_memory":
            content = payload.pop("content")
            kwargs = payload.pop("kwargs", {})
            # Use pre-computed embedding if present, otherwise generate
            emb = payload.pop("embedding", None)
            if emb is None:
                emb = embeddings.embed_text(content)
            db.add_memory(self.conn, content=content, embedding=emb, **kwargs)

        elif op == "update_memory":
            db.update_memory(self.conn, **payload)

        elif op == "archive_memory":
            db.archive_memory(self.conn, **payload)

        else:
            raise ValueError(f"Unknown write operation: {op}")

    # -- diagnostics ---------------------------------------------------------

    @property
    def pending_count(self) -> int:
        """Number of pending writes in the queue."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM write_queue WHERE status = 'pending'"
        ).fetchone()
        return row[0] if row else 0

    @property
    def stats(self) -> dict[str, int]:
        """Cumulative worker stats (processed / failed)."""
        with self._lock:
            return dict(self._stats)


# ---------------------------------------------------------------------------
# Module-level convenience functions (no worker pool required)
# ---------------------------------------------------------------------------

def enqueue(conn, operation: str, payload: dict, priority: int = 3) -> int:
    """Enqueue a write operation without a running WriteQueue.

    The write is persisted to the write_queue table and can be processed
    later by a WriteQueue instance or by calling ``flush()``.
    """
    return db.enqueue_write(conn, operation, json.dumps(payload), priority)


def flush(conn, batch_size: int = 50) -> dict[str, int]:
    """Force-process all pending writes immediately (synchronous).

    Iterates through pending writes in priority order, processes each one,
    and returns counts of processed and failed writes.
    """
    processed = 0
    failed = 0

    while True:
        pending = db.get_pending_writes(conn, limit=batch_size)
        if not pending:
            break

        for write in pending:
            try:
                _process_write_standalone(conn, write)
                db.mark_write_complete(
                    conn, write["id"], status="completed"
                )
                processed += 1
            except Exception as exc:
                db.mark_write_complete(
                    conn, write["id"],
                    status="failed", error=str(exc),
                )
                failed += 1
                log.error("flush: write %d failed: %s", write["id"], exc)

    return {"processed": processed, "failed": failed}


def _process_write_standalone(conn, write: dict):
    """Process a single write operation (standalone, no WriteQueue instance)."""
    payload = json.loads(write["payload"])
    op = write["operation"]

    if op == "add_memory":
        content = payload.pop("content")
        kwargs = payload.pop("kwargs", {})
        emb = payload.pop("embedding", None)
        if emb is None:
            emb = embeddings.embed_text(content)
        db.add_memory(conn, content=content, embedding=emb, **kwargs)

    elif op == "update_memory":
        db.update_memory(conn, **payload)

    elif op == "archive_memory":
        db.archive_memory(conn, **payload)

    else:
        raise ValueError(f"Unknown write operation: {op}")
