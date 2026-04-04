"""FadeMem decay scoring -- importance-weighted memory lifecycle management.

Implements a 4-weight importance function that balances relevance, frequency,
recency, and surprise.  Also provides Weibull-style decay and archival logic
that respects governance layers.
"""

from __future__ import annotations

import math
from typing import Optional, Set

from .hierarchy import LAYER_CONFIGS, get_layer_config

# ---------------------------------------------------------------------------
# Tags whose presence protects a memory from archival regardless of score
# ---------------------------------------------------------------------------
PROTECTED_TAGS: Set[str] = frozenset({
    "correction",
    "decision",
    "identity",
    "emotional_anchor",
    "commitment",
    "exact_value",
    "chain_of_command",
    "person",
})


def importance_score(
    relevance: float,
    access_count: int,
    max_access_count: int,
    hours_since_access: float,
    surprise_score: float = 0.5,
    decay_rate: float = 0.1,
    alpha: float = 0.30,
    beta: float = 0.25,
    gamma: float = 0.25,
    delta: float = 0.20,
) -> float:
    """Compute the FadeMem importance score I(t).

    I(t) = alpha*relevance + beta*frequency + gamma*recency + delta*surprise

    Parameters
    ----------
    relevance : float
        Semantic relevance score in [0, 1].
    access_count : int
        How many times this memory has been accessed.
    max_access_count : int
        Maximum access count across all memories (for normalization).
    hours_since_access : float
        Hours since the memory was last accessed.
    surprise_score : float
        Titans-inspired surprise value in [0, 1].
    decay_rate : float
        Exponential decay rate for the recency term.
    alpha, beta, gamma, delta : float
        Weights for the four components (should sum to 1.0).
    """
    # Frequency: log-normalized
    max_ac = max(max_access_count, 1)
    frequency = math.log(access_count + 1) / math.log(max_ac + 1) if max_ac > 0 else 0.0

    # Recency: exponential decay
    recency = math.exp(-decay_rate * max(hours_since_access, 0.0))

    # Clamp inputs
    relevance = max(min(relevance, 1.0), 0.0)
    surprise_score = max(min(surprise_score, 1.0), 0.0)

    I = alpha * relevance + beta * frequency + gamma * recency + delta * surprise_score
    return max(min(I, 1.0), 0.0)


def decay_value(
    initial_strength: float,
    hours_since_access: float,
    decay_rate: float = 0.1,
    beta: float = 0.8,
) -> float:
    """Compute decayed memory strength using a Weibull-style function.

    v(t) = v(0) * exp(-lambda * t^beta)

    Parameters
    ----------
    initial_strength : float
        Original importance/strength value.
    hours_since_access : float
        Hours since last access.
    decay_rate : float
        Lambda -- controls how fast the memory decays.
    beta : float
        Shape parameter.  <1 = fast-then-slow, >1 = slow-then-fast.
    """
    t = max(hours_since_access, 0.0)
    try:
        return initial_strength * math.exp(-decay_rate * (t ** beta))
    except OverflowError:
        return 0.0


def should_archive(
    importance: float,
    age_days: float,
    tags: Optional[Set[str]] = None,
    governance_layer: int = 3,
) -> bool:
    """Decide whether a memory should be archived (sent to the graveyard).

    Rules:
    - Memories with protected tags are NEVER archived.
    - Layer 1 and 2 memories are NEVER archived.
    - Layer 3: archive if importance < 0.1 and age > 30 days.
    - Layer 4: archive if importance < 0.15 and age > 14 days.
    """
    tags = tags or set()

    # Protected tags override everything
    if tags & PROTECTED_TAGS:
        return False

    layer = get_layer_config(governance_layer)

    if not layer.can_archive:
        return False

    if layer.archive_threshold is None or layer.min_archive_age_days is None:
        return False

    return importance < layer.archive_threshold and age_days > layer.min_archive_age_days


def get_memory_layer(importance: float) -> str:
    """Classify a memory's storage tier based on current importance.

    Returns one of: "LTM" (long-term), "STM" (short-term), or "current".

    Thresholds:
    - importance >= 0.7  ->  promote to LTM
    - importance >= 0.3  ->  STM
    - importance < 0.3   ->  current (candidate for demotion/archive)
    """
    if importance >= 0.7:
        return "LTM"
    elif importance >= 0.3:
        return "STM"
    else:
        return "current"
