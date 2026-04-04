"""
memory-v3 -- Async Memory Operations

Priority-based write queue with background workers, and batched embedding
generation for high-throughput memory ingestion.

Classes:
    WriteQueue -- background worker pool that processes queued writes

Functions:
    enqueue   -- add a write operation to the queue (module-level convenience)
    flush     -- force-process all pending writes immediately
    batch_embed_queued -- embed all pending add_memory writes in batches
"""

from .write_queue import WriteQueue, enqueue, flush
from .batch_embedder import batch_embed_queued

__all__ = [
    "WriteQueue",
    "enqueue",
    "flush",
    "batch_embed_queued",
]
