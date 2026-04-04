"""
memory-v3 -- Multi-Graph Knowledge Graph Builder

Extracts entities and relationships from vault markdown files and routes
them to the 4 graph layers (semantic, temporal, causal, entity) via
GraphManager.add_extracted(). Incremental with file hashing.

Uses the local LLM (Ollama) for entity/relationship extraction.
"""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ollama

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
# LLM extraction
# ---------------------------------------------------------------------------

def _extract_from_text(text: str, model: str) -> dict:
    """Use LLM to extract entities and relationships from text.
    Returns {"entities": [...], "relationships": [...]}."""
    # Truncate long texts
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

    # Parse JSON from LLM response
    try:
        # Try to find JSON block in the response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(content)

        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        # Validate structure
        valid_entities = []
        for e in entities:
            if isinstance(e, dict) and e.get("name"):
                valid_entities.append({
                    "name": str(e["name"]).strip(),
                    "type": str(e.get("type", "concept")).strip().lower(),
                    "properties": e.get("properties", {}),
                })

        valid_rels = []
        for r in relationships:
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
) -> dict:
    """Build multi-graph knowledge graph from vault markdown files.

    Incremental: only processes files whose hash has changed since last build.
    Set force=True to rebuild everything.

    Args:
        vault_path: Path to vault directory. Uses config if None.
        force: Force full rebuild ignoring file hashes.
        conn: Database connection. Creates one if None.

    Returns:
        Stats dict with per-layer counts and file processing summary.
    """
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
        "files_processed": 0,
        "files_skipped": 0,
        "files_errored": 0,
        "total_entities": 0,
        "total_relationships": 0,
        "per_layer": {},
    }

    # Collect all markdown files
    md_files = sorted(vault.rglob("*.md"))

    for md_file in md_files:
        rel_path = str(md_file.relative_to(vault)).replace("\\", "/")

        # Skip changelog and manifest
        if rel_path in ("changelog.md", "manifest.json"):
            continue

        try:
            content_hash = _file_hash(md_file)
        except Exception:
            stats["files_errored"] += 1
            continue

        # Incremental check: skip if hash unchanged
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

        # Read file content
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception:
            stats["files_errored"] += 1
            continue

        # Skip very short files
        if len(text.strip()) < 50:
            stats["files_skipped"] += 1
            continue

        # Extract entities and relationships via LLM
        extracted = _extract_from_text(text, model)

        entities = extracted.get("entities", [])
        relationships = extracted.get("relationships", [])

        if not entities and not relationships:
            # Nothing extracted, but still mark as processed
            _update_kg_hash(conn, rel_path, content_hash, now_iso, 0)
            stats["files_skipped"] += 1
            continue

        # Route to graph layers via GraphManager
        try:
            layer_stats = gm.add_extracted(
                entities=entities,
                relationships=relationships,
                source_file=rel_path,
                timestamp=now_iso,
            )
        except Exception as e:
            stats["files_errored"] += 1
            continue

        # Accumulate per-layer stats
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

        # Update file hash for incremental tracking
        _update_kg_hash(
            conn, rel_path, content_hash, now_iso,
            len(entities) + len(relationships),
        )

    # Save all graphs to disk
    try:
        gm.save_all()
    except Exception:
        pass

    return stats


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
