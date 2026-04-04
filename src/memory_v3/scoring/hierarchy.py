"""Constitutional memory hierarchy -- governance layers with per-layer scoring configs.

Each memory belongs to one of four governance layers that determine its decay,
activation floor, archival policy, and overall durability. Higher layers are
more protected; lower layers are more ephemeral.

Layer 1 - Constitutional: identity, chain-of-command. Never decays, never archives.
Layer 2 - Legislative: decisions, corrections, commitments. Minimal decay, no archive.
Layer 3 - Factual: facts, references, tools. Standard decay, archivable after 30 days.
Layer 4 - Ephemeral: episodes, conversations. Fast decay, archivable after 14 days.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set


@dataclass(frozen=True)
class LayerConfig:
    """Scoring and lifecycle configuration for a single governance layer."""

    name: str
    level: int
    decay_rate: float
    activation_floor: float
    fademem_beta: Optional[float]
    archive_threshold: Optional[float]
    min_archive_age_days: Optional[int]
    can_archive: bool


# ---------------------------------------------------------------------------
# Layer configuration table
# ---------------------------------------------------------------------------

LAYER_CONFIGS: Dict[int, LayerConfig] = {
    1: LayerConfig(
        name="Constitutional",
        level=1,
        decay_rate=0.0,
        activation_floor=5.0,
        fademem_beta=None,
        archive_threshold=None,
        min_archive_age_days=None,
        can_archive=False,
    ),
    2: LayerConfig(
        name="Legislative",
        level=2,
        decay_rate=0.01,
        activation_floor=2.0,
        fademem_beta=0.5,
        archive_threshold=None,
        min_archive_age_days=None,
        can_archive=False,
    ),
    3: LayerConfig(
        name="Factual",
        level=3,
        decay_rate=0.10,
        activation_floor=-0.5,
        fademem_beta=0.8,
        archive_threshold=0.1,
        min_archive_age_days=30,
        can_archive=True,
    ),
    4: LayerConfig(
        name="Ephemeral",
        level=4,
        decay_rate=0.25,
        activation_floor=-2.0,
        fademem_beta=1.2,
        archive_threshold=0.15,
        min_archive_age_days=14,
        can_archive=True,
    ),
}

# Content types that map to specific layers
_LAYER_1_TYPES: Set[str] = {"identity", "chain_of_command"}
_LAYER_2_TYPES: Set[str] = {"decision", "correction", "commitment"}
_LAYER_3_TYPES: Set[str] = {"fact", "reference", "tool"}
_LAYER_4_TYPES: Set[str] = {"episode", "conversation"}

_LAYER_1_TAGS: Set[str] = {"identity", "chain_of_command"}
_LAYER_2_TAGS: Set[str] = {"decision", "correction", "commitment", "exact_value"}


def get_layer_config(governance_layer: int) -> LayerConfig:
    """Return the LayerConfig for the given governance layer (1-4).

    Falls back to Layer 3 (Factual) if the layer number is unknown.
    """
    return LAYER_CONFIGS.get(governance_layer, LAYER_CONFIGS[3])


def classify_governance_layer(
    content_type: str,
    protected: bool = False,
    tags: Optional[Set[str]] = None,
) -> int:
    """Automatically classify a memory into a governance layer.

    Priority order:
    1. Tag-based override (identity/chain_of_command tags -> Layer 1)
    2. Content type mapping
    3. Protected flag bump (protected memories get at least Layer 2)
    4. Default to Layer 3 (Factual)
    """
    tags = tags or set()
    ct = content_type.lower().strip() if content_type else ""

    # Tag-based overrides take highest priority
    if tags & _LAYER_1_TAGS:
        return 1
    if tags & _LAYER_2_TAGS:
        return 2

    # Content-type mapping
    if ct in _LAYER_1_TYPES:
        return 1
    if ct in _LAYER_2_TYPES:
        return 2
    if ct in _LAYER_4_TYPES:
        return 4
    if ct in _LAYER_3_TYPES:
        return 3

    # Protected memories get at least Layer 2
    if protected:
        return 2

    # Default
    return 3
