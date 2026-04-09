"""
memory-v3 -- Multi-Graph Knowledge Graph Builder

Extracts entities and relationships from vault markdown files and routes
them to the 4 graph layers (semantic, temporal, causal, entity) via
GraphManager.add_extracted(). Incremental with file hashing.

Supports two LLM backends:
  - "ollama" (default): local Ollama for incremental updates
  - "anthropic": Anthropic API (Haiku) with async batching for bulk builds
"""

import asyncio
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import get_config
from .db import init_db

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """Extract entities and relationships from this text.

Return JSON with two lists:
1. "entities": each with name, type (concept/tool/person/team/event/error/fix/decision/milestone), properties (optional dict)
2. "relationships": each with source, target, edge_type, description (optional)

Valid edge_types:
- Semantic: related_to, uses, part_of, implements, depends_on, similar_to
- Temporal: preceded_by, followed_by, concurrent_with, same_session
- Causal: caused, enabled, prevented, motivated, triggered_by, resolved
- Entity: built, decided, maintains, member_of, responsible_for

Rules:
- Extract concrete entities (named things, not generic nouns)
- Relationships should connect two extracted entities
- Prefer specific edge_types over generic "related_to"
- If nothing extractable, return {{"entities": [], "relationships": []}}

Text:
{text}

JSON:"""


# ---------------------------------------------------------------------------
# Response parser (shared by both backends)
# ---------------------------------------------------------------------------

def _parse_extraction(content: str) -> dict:
    """Parse LLM response into validated entities and relationships."""
    try:
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(content)

        valid_entities = []
        for e in data.get("entities", []):
            if isinstance(e, dict) and e.get("name"):
                valid_entities.append({
                    "name": str(e["name"]).strip(),
                    "type": str(e.get("type", "concept")).strip().lower(),
                    "properties": e.get("properties", {}),
                })

        valid_rels = []
        for r in data.get("relationships", []):
            if isinstance(r, dict) and r.get("source") and r.get("target"):
                valid_rels.append({
                    "source": str(r["source"]).strip(),
                    "target": str(r["target"]).strip(),
                    "edge_type": str(r.get("edge_type", "related_to")).strip().lower(),
                    "description": str(r.get("description", "")),
                    "confidence": float(r.get("confidence", 0.8)),
                })

        return {"entities": valid_entities, "relationships": valid_rels}

    except (json.JSONDecodeError, TypeError, ValueError):
        return {"entities": [], "relationships": [], "parse_error": content[:200]}


# ---------------------------------------------------------------------------
# Ollama backend (synchronous, for incremental)
# ---------------------------------------------------------------------------

def _extract_ollama(text: str, model: str) -> dict:
    """Extract via local Ollama."""
    import ollama

    if len(text) > 4000:
        text = text[:4000]

    prompt = EXTRACT_PROMPT.format(text=text)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 2048},
        )
        content = response["message"]["content"].strip()
    except Exception as e:
        return {"entities": [], "relationships": [], "error": str(e)}

    return _parse_extraction(content)


# ---------------------------------------------------------------------------
# Anthropic backend (async batched, for bulk builds)
# ---------------------------------------------------------------------------

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_MAX_CONCURRENT = 20  # parallel API calls
ANTHROPIC_BATCH_SIZE = 50      # files per batch before committing


async def _extract_anthropic_single(client, text: str, semaphore) -> dict:
    """Single Anthropic API call with semaphore for rate limiting."""
    if len(text) > 4000:
        text = text[:4000]

    prompt = EXTRACT_PROMPT.format(text=text)

    async with semaphore:
        try:
            response = await client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=2048,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()
            return _parse_extraction(content)
        except Exception as e:
            return {"entities": [], "relationships": [], "error": str(e)}


async def _extract_anthropic_batch(
    client, file_texts: list[tuple[str, str, str]],
    semaphore,
) -> list[tuple[str, str, dict]]:
    """Extract from a batch of (rel_path, content_hash, text) tuples.
    Returns list of (rel_path, content_hash, extraction_result)."""
    tasks = []
    for rel_path, content_hash, text in file_texts:
        task = _extract_anthropic_single(client, text, semaphore)
        tasks.append((rel_path, content_hash, task))

    results = []
    for rel_path, content_hash, task in tasks:
        extraction = await task
        results.append((rel_path, content_hash, extraction))

    return results


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_knowledge_graph(
    vault_path: Optional[str] = None,
    force: bool = False,
    conn=None,
    backend: str = "ollama",
) -> dict:
    """Build multi-graph knowledge graph from vault markdown files.

    Args:
        vault_path: Path to vault directory. Uses config if None.
        force: Force full rebuild ignoring file hashes.
        conn: Database connection. Creates one if None.
        backend: "ollama" for local LLM, "anthropic" for Anthropic API.

    Returns:
        Stats dict with per-layer counts and file processing summary.
    """
    if backend == "anthropic":
        return asyncio.run(_build_kg_anthropic(vault_path, force, conn))
    else:
        return _build_kg_ollama(vault_path, force, conn)


def _build_kg_ollama(
    vault_path: Optional[str] = None,
    force: bool = False,
    conn=None,
) -> dict:
    """Original synchronous Ollama-based builder."""
    cfg = get_config()
    vault_path = vault_path or cfg.vault_path

    if not vault_path or not os.path.isdir(vault_path):
        return {"error": "No vault path configured or directory not found", "vault_path": vault_path}

    if conn is None:
        conn = init_db()

    from .graphs import GraphManager
    gm = GraphManager(graph_dir=cfg.graph_dir)

    vault = Path(vault_path)
    model = cfg.llm_model
    now_iso = datetime.now(timezone.utc).isoformat()

    stats = {
        "backend": "ollama",
        "files_processed": 0,
        "files_skipped": 0,
        "files_errored": 0,
        "total_entities": 0,
        "total_relationships": 0,
        "per_layer": {},
    }

    md_files = sorted(vault.rglob("*.md"))

    for md_file in md_files:
        rel_path = str(md_file.relative_to(vault)).replace("\\", "/")

        if rel_path in ("changelog.md", "manifest.json"):
            continue

        try:
            content_hash = _file_hash(md_file)
        except Exception:
            stats["files_errored"] += 1
            continue

        if not force:
            row = conn.execute(
                "SELECT content_hash FROM file_hashes WHERE file_path = ?",
                (f"kg:{rel_path}",),
            ).fetchone()
            existing_hash = None
            if row:
                existing_hash = row["content_hash"] if isinstance(row, dict) else row[0]
            if existing_hash == content_hash:
                stats["files_skipped"] += 1
                continue

        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception:
            stats["files_errored"] += 1
            continue

        if len(text.strip()) < 50:
            stats["files_skipped"] += 1
            continue

        extracted = _extract_ollama(text, model)
        _process_extraction(
            conn, gm, extracted, rel_path, content_hash, now_iso, stats,
        )

    try:
        gm.save_all()
    except Exception:
        pass

    return stats


async def _build_kg_anthropic(
    vault_path: Optional[str] = None,
    force: bool = False,
    conn=None,
) -> dict:
    """Async Anthropic-based builder with batched concurrent calls."""
    import anthropic

    cfg = get_config()
    vault_path = vault_path or cfg.vault_path

    if not vault_path or not os.path.isdir(vault_path):
        return {"error": "No vault path configured or directory not found", "vault_path": vault_path}

    if conn is None:
        conn = init_db()

    from .graphs import GraphManager
    gm = GraphManager(graph_dir=cfg.graph_dir)

    vault = Path(vault_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    stats = {
        "backend": "anthropic",
        "model": ANTHROPIC_MODEL,
        "files_processed": 0,
        "files_skipped": 0,
        "files_errored": 0,
        "total_entities": 0,
        "total_relationships": 0,
        "per_layer": {},
        "api_calls": 0,
        "elapsed_seconds": 0,
    }

    # Collect files that need processing
    pending_files = []
    md_files = sorted(vault.rglob("*.md"))

    for md_file in md_files:
        rel_path = str(md_file.relative_to(vault)).replace("\\", "/")

        if rel_path in ("changelog.md", "manifest.json"):
            continue

        try:
            content_hash = _file_hash(md_file)
        except Exception:
            stats["files_errored"] += 1
            continue

        if not force:
            row = conn.execute(
                "SELECT content_hash FROM file_hashes WHERE file_path = ?",
                (f"kg:{rel_path}",),
            ).fetchone()
            existing_hash = None
            if row:
                existing_hash = row["content_hash"] if isinstance(row, dict) else row[0]
            if existing_hash == content_hash:
                stats["files_skipped"] += 1
                continue

        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception:
            stats["files_errored"] += 1
            continue

        if len(text.strip()) < 50:
            stats["files_skipped"] += 1
            continue

        pending_files.append((rel_path, content_hash, text))

    total = len(pending_files)
    print(f"[build-kg] {total} files to process via Anthropic ({ANTHROPIC_MODEL})", flush=True)
    print(f"[build-kg] {stats['files_skipped']} skipped, {stats['files_errored']} errored during scan", flush=True)

    if not pending_files:
        return stats

    # Process in batches with async concurrency
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(ANTHROPIC_MAX_CONCURRENT)
    start_time = time.time()

    for batch_start in range(0, total, ANTHROPIC_BATCH_SIZE):
        batch = pending_files[batch_start:batch_start + ANTHROPIC_BATCH_SIZE]
        batch_num = batch_start // ANTHROPIC_BATCH_SIZE + 1
        total_batches = (total + ANTHROPIC_BATCH_SIZE - 1) // ANTHROPIC_BATCH_SIZE

        # Fire all calls in this batch concurrently (limited by semaphore)
        tasks = []
        for rel_path, content_hash, text in batch:
            task = asyncio.create_task(
                _extract_anthropic_single(client, text, semaphore)
            )
            tasks.append((rel_path, content_hash, task))

        # Await all results
        for rel_path, content_hash, task in tasks:
            extracted = await task
            stats["api_calls"] += 1
            _process_extraction(
                conn, gm, extracted, rel_path, content_hash, now_iso, stats,
            )

        elapsed = time.time() - start_time
        rate = stats["api_calls"] / elapsed if elapsed > 0 else 0
        print(
            f"[build-kg] Batch {batch_num}/{total_batches} done | "
            f"{stats['files_processed']} processed | "
            f"{stats['api_calls']} API calls | "
            f"{rate:.1f} calls/sec | "
            f"{elapsed:.0f}s elapsed",
            flush=True,
        )

    stats["elapsed_seconds"] = round(time.time() - start_time, 1)

    try:
        gm.save_all()
    except Exception:
        pass

    await client.close()

    print(f"[build-kg] COMPLETE: {stats['files_processed']} files, "
          f"{stats['total_entities']} entities, "
          f"{stats['total_relationships']} relationships, "
          f"{stats['elapsed_seconds']}s", flush=True)

    return stats


# ---------------------------------------------------------------------------
# Shared extraction processor
# ---------------------------------------------------------------------------

def _process_extraction(
    conn, gm, extracted: dict, rel_path: str, content_hash: str,
    now_iso: str, stats: dict,
):
    """Process an extraction result: route to graphs and update stats."""
    entities = extracted.get("entities", [])
    relationships = extracted.get("relationships", [])

    if not entities and not relationships:
        _update_kg_hash(conn, rel_path, content_hash, now_iso, 0)
        stats["files_skipped"] += 1
        return

    try:
        layer_stats = gm.add_extracted(
            entities=entities,
            relationships=relationships,
            source_file=rel_path,
            timestamp=now_iso,
        )
    except Exception:
        stats["files_errored"] += 1
        return

    for layer_name, ls in layer_stats.items():
        if layer_name not in stats["per_layer"]:
            stats["per_layer"][layer_name] = {
                "nodes_added": 0, "nodes_updated": 0, "edges_added": 0,
            }
        for key in ("nodes_added", "nodes_updated", "edges_added"):
            stats["per_layer"][layer_name][key] += ls.get(key, 0)

    stats["total_entities"] += len(entities)
    stats["total_relationships"] += len(relationships)
    stats["files_processed"] += 1

    _update_kg_hash(
        conn, rel_path, content_hash, now_iso,
        len(entities) + len(relationships),
    )


def _update_kg_hash(conn, rel_path: str, content_hash: str, timestamp: str, chunk_count: int):
    """Update the file hash for knowledge graph incremental tracking."""
    conn.execute(
        """INSERT INTO file_hashes (file_path, content_hash, indexed_at, chunk_count)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            content_hash = excluded.content_hash,
            indexed_at = excluded.indexed_at,
            chunk_count = excluded.chunk_count""",
        (f"kg:{rel_path}", content_hash, timestamp, chunk_count),
    )
    conn.commit()
