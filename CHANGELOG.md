# Changelog

All notable changes to memory-v3 are documented in this file.

## [3.0.0] - 2026-04-04

Initial release of memory-v3, the third generation of the memory system.
Predecessors: [memory-v2](https://github.com/your-org/memory-v2), [memory-v1](https://github.com/your-org/memory-v1).

### 12 Upgrades from v2

1. **Constitutional Memory Hierarchy** -- 4 governance layers (Constitutional, Legislative, Factual, Ephemeral) with per-layer decay rates, activation floors, and archival policies.

2. **Titans-Inspired Surprise Scoring** -- Novelty detection via centroid-based clustering. High-surprise memories receive a scoring boost to ensure novel information is retained.

3. **Sensory Buffer (Pre-Encoding Gate)** -- New `sensory_buffer` table acts as an intake queue. Incoming facts are evaluated for novelty and deduplication before being committed to memory.

4. **Async Write Queue** -- New `write_queue` table decouples memory writes from the request path, enabling non-blocking ingestion with priority ordering.

5. **Action Log** -- New `action_log` table records every extraction decision (ADD/UPDATE/DELETE/NONE) with confidence scores and model version, providing a full audit trail.

6. **Consolidation Log** -- New `consolidation_log` table tracks merge/split/prune cycles for memory maintenance.

7. **Multi-Graph Architecture** -- Separate semantic, temporal, causal, and entity graphs with a query router that analyzes intent to select the right layer(s).

8. **FadeMem 4-Weight Importance** -- Extended importance function: `I(t) = alpha*relevance + beta*frequency + gamma*recency + delta*surprise`. The surprise component (delta) is new in v3.

9. **HaluMem Confidence Decay** -- Unverified memories gradually lose confidence over time, with `verification_status` and `last_verified_at` tracking.

10. **Zettelkasten Linking** -- `linked_memories` and `link_descriptions` columns enable explicit bidirectional links between memories with typed descriptions.

11. **Structure Tags** -- `structure_type` column classifies memories as narrative, atomic, list, or procedural for retrieval-time formatting.

12. **Schema Version Tracking** -- New `schema_version` table records migration history with timestamps and notes.

### New Tables (6)

- `sensory_buffer` -- pre-encoding intake queue
- `cluster_centroids` -- centroid embeddings for surprise scoring
- `action_log` -- extraction decision audit trail
- `consolidation_log` -- merge/split/prune cycle history
- `write_queue` -- async write operations with priority
- `schema_version` -- migration version tracking

### New Columns on `memories` (12)

- `governance_layer` -- governance tier (1-4)
- `confidence_floor` -- minimum confidence threshold
- `last_verified_at` -- timestamp of last verification
- `verification_status` -- unverified/verified/disputed
- `cross_ref_count` -- number of cross-references from other memories
- `structure_type` -- narrative/atomic/list/procedural
- `surprise_score` -- Titans-inspired novelty score (0-1)
- `memory_layer` -- STM/LTM/current storage tier
- `cluster_id` -- assigned cluster for consolidation
- `linked_memories` -- JSON array of linked memory IDs
- `link_descriptions` -- JSON object of link type descriptions
- `source_hash` -- hash of the original source content

### Migration

Run `memory-v3-migrate --db /path/to/v2.db` to upgrade a v2 database in-place.
The migration is idempotent and safe to run multiple times.
