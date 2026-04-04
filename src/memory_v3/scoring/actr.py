"""ACT-R activation scoring for memory retrieval.

Implements the Adaptive Control of Thought -- Rational (ACT-R) framework for
computing memory activation levels.  Higher activation means the memory is
more likely to be needed in the current context.

Key equations:
  Base-level:  B_i = ln(n / (1 - d)) - d * ln(L)
  Spreading:   S_i = sum(W_j * (S_max - ln(fan_j)))
  Noise:       epsilon ~ Logistic(0, s)
  Activation:  A_i = B_i + S_i + epsilon
  Retrieval P: P = 1 / (1 + exp(-(A - tau) / s))
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import numpy as np

# ---------------------------------------------------------------------------
# Constants (local defaults -- config module can override)
# ---------------------------------------------------------------------------
DECAY_D: float = 0.5
S_MAX: float = 1.6
NOISE_S: float = 0.25
RETRIEVAL_TAU: float = -0.5


def _hours_since(dt: datetime) -> float:
    """Return hours elapsed since *dt* (assumed UTC if naive)."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    return max(delta.total_seconds() / 3600.0, 0.001)


# ---------------------------------------------------------------------------
# Core ACT-R functions
# ---------------------------------------------------------------------------

def base_level_activation(
    access_count: int,
    created_at: datetime,
    last_accessed_at: datetime,
    decay: float = DECAY_D,
) -> float:
    """Compute base-level activation B_i.

    B_i = ln(n / (1 - d)) - d * ln(L)

    where *n* is access count and *L* is the memory's lifetime in hours.
    """
    n = max(access_count, 1)
    L = _hours_since(created_at)
    try:
        B = math.log(n / (1.0 - decay)) - decay * math.log(L)
    except (ValueError, ZeroDivisionError):
        B = 0.0
    return B


def spreading_activation(
    memory_tags: Set[str],
    context_tags: Set[str],
    tag_fan: Dict[str, int],
) -> float:
    """Compute spreading activation S_i from shared tags.

    S_i = sum over shared tags j of: W_j * (S_max - ln(fan_j))

    W_j is set to 1/|context_tags| so total source activation sums to 1.
    """
    if not context_tags or not memory_tags:
        return 0.0

    shared = memory_tags & context_tags
    if not shared:
        return 0.0

    W = 1.0 / len(context_tags)
    S = 0.0
    for tag in shared:
        fan = max(tag_fan.get(tag, 1), 1)
        strength = S_MAX - math.log(fan)
        S += W * strength

    return S


def activation_noise(s: float = NOISE_S) -> float:
    """Sample activation noise from a logistic distribution.

    The logistic distribution with location 0 and scale s has variance
    (pi * s)^2 / 3, providing realistic variability in retrieval.
    """
    # Clamp the uniform sample away from 0 and 1 to avoid log domain errors
    u = random.random()
    u = max(min(u, 0.9999), 0.0001)
    return s * math.log(u / (1.0 - u))


def full_activation(
    access_count: int,
    created_at: datetime,
    last_accessed_at: datetime,
    memory_tags: Set[str],
    context_tags: Set[str],
    tag_fan: Dict[str, int],
    protected: bool = False,
    activation_floor: float = -0.5,
) -> float:
    """Compute full ACT-R activation with noise, floored.

    A_i = max(B_i + S_i + epsilon, activation_floor)

    Protected memories get an activation floor of at least 0.0.
    """
    B = base_level_activation(access_count, created_at, last_accessed_at)
    S = spreading_activation(memory_tags, context_tags, tag_fan)
    noise = activation_noise()

    floor = max(activation_floor, 0.0) if protected else activation_floor
    A = max(B + S + noise, floor)
    return A


def retrieval_probability(
    activation: float,
    tau: float = RETRIEVAL_TAU,
    s: float = NOISE_S,
) -> float:
    """Compute retrieval probability using the softmax/sigmoidal rule.

    P = 1 / (1 + exp(-(A - tau) / s))
    """
    try:
        exponent = -(activation - tau) / s
        exponent = max(min(exponent, 500), -500)  # clamp to avoid overflow
        return 1.0 / (1.0 + math.exp(exponent))
    except OverflowError:
        return 0.0 if activation < tau else 1.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two numpy vectors."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
