"""Fan effect penalty from ACT-R + Embeddings (Frontiers 2026).

The fan effect models the cognitive finding that as an item becomes
associated with more facts (higher "fan"), each individual association
becomes harder to retrieve.  This module adds a logarithmic penalty
on top of the standard spreading activation calculation.

penalty(n) = delta * ln(n + 1)
"""

from __future__ import annotations

import math
from typing import Dict, Set

from .actr import S_MAX


def fan_penalty(total_connections: int, delta: float = 0.3) -> float:
    """Compute the fan effect penalty for a tag with *total_connections*.

    penalty = delta * ln(total_connections + 1)

    A tag connected to many memories incurs a larger penalty, reducing
    its contribution to spreading activation.
    """
    return delta * math.log(max(total_connections, 0) + 1)


def spreading_activation_with_fan(
    memory_tags: Set[str],
    context_tags: Set[str],
    tag_fan: Dict[str, int],
    fan_delta: float = 0.3,
) -> float:
    """Spreading activation with per-tag fan penalty.

    S_i = sum over shared tags j of:
        W_j * max(S_max - ln(fan_j) - fan_penalty(total_connections_j, delta), 0)

    This is identical to the standard ACT-R spreading activation except
    each tag's strength is further reduced by the fan penalty, and the
    per-tag contribution is floored at zero (cannot go negative).
    """
    if not context_tags or not memory_tags:
        return 0.0

    shared = memory_tags & context_tags
    if not shared:
        return 0.0

    W = 1.0 / len(context_tags)
    S = 0.0

    for tag in shared:
        fan_count = max(tag_fan.get(tag, 1), 1)
        penalty = fan_penalty(fan_count, delta=fan_delta)
        strength = S_MAX - math.log(fan_count) - penalty
        S += W * max(strength, 0.0)

    return S
