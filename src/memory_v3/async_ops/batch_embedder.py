"""
memory-v3 -- Batched Embedding Generator

Scans the write_queue for pending ``add_memory`` operations that lack
pre-computed embeddings, generates embeddings in batches via
``embeddings.embed_batch()``, and updates the payloads in-place so the
write workers can skip the embedding step.

This is useful when bulk-ingesting many memories: queue them all first
without embeddings, then call ``batch_embed_queued()`` once to embed
everything in efficient batches before flushing the queue.
"""

import json
import logging

from .. import db, embeddings

log = logging.getLogger(__name__)


def batch_embed_queued(conn, batch_size: int = 10) -> dict:
    """Embed all pending add_memory writes that need embeddings, in batches.

    Scans the write_queue for pending ``add_memory`` operations whose
    payloads do not already contain an ``embedding`` key.  Embeds the
    content texts in batches and writes the embeddings back into the
    queue payloads.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    batch_size : int
        Number of texts to embed per Ollama call.

    Returns
    -------
    dict
        Statistics: ``embedded``, ``skipped``, ``failed``, ``batches``.
    """
    # Fetch all pending add_memory writes
    rows = conn.execute(
        """SELECT id, payload FROM write_queue
        WHERE status = 'pending' AND operation = 'add_memory'
        ORDER BY priority ASC, created_at ASC"""
    ).fetchall()

    # Filter to those missing an embedding
    needs_embedding = []
    skipped = 0
    for row in rows:
        payload = json.loads(row["payload"])
        if "embedding" in payload and payload["embedding"] is not None:
            skipped += 1
            continue
        needs_embedding.append({
            "id": row["id"],
            "payload": payload,
            "content": payload.get("content", ""),
        })

    if not needs_embedding:
        return {
            "embedded": 0,
            "skipped": skipped,
            "failed": 0,
            "batches": 0,
        }

    # Process in batches
    embedded = 0
    failed = 0
    batches = 0

    for i in range(0, len(needs_embedding), batch_size):
        batch = needs_embedding[i : i + batch_size]
        texts = [item["content"] for item in batch]

        try:
            vectors = embeddings.embed_batch(texts)
            batches += 1
        except Exception as exc:
            log.error("Batch embedding failed for batch %d: %s", i // batch_size, exc)
            failed += len(batch)
            continue

        # Write embeddings back into queue payloads
        for item, vector in zip(batch, vectors):
            try:
                item["payload"]["embedding"] = vector
                conn.execute(
                    "UPDATE write_queue SET payload = ? WHERE id = ?",
                    (json.dumps(item["payload"]), item["id"]),
                )
                embedded += 1
            except Exception as exc:
                log.error("Failed to update payload for write %d: %s", item["id"], exc)
                failed += 1

    conn.commit()

    log.info(
        "batch_embed_queued: %d embedded, %d skipped, %d failed, %d batches",
        embedded, skipped, failed, batches,
    )
    return {
        "embedded": embedded,
        "skipped": skipped,
        "failed": failed,
        "batches": batches,
    }
