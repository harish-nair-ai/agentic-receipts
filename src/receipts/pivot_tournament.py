"""Probabilistic Pivot Tournament — best-of-N selection in O(N·k).

Ported from the base paper's reference repo. Pairwise Bradley-Terry comparisons against
a small set of pivots avoid the instability of pointwise scores while staying linear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_PIVOTS = 2


@dataclass
class Candidate:
    """A ranking candidate with a calibrated 0..1 rating."""

    id: str
    rating: float


def bradley_terry_prob(ra: float, rb: float) -> float:
    """P(a beats b) = 1 / (1 + exp(-(ra - rb)))."""
    return 1.0 / (1.0 + math.exp(-(ra - rb)))


def comparison_count(n: int, pivots: int) -> int:
    """Number of pairwise comparisons performed for n candidates and `pivots` pivots."""
    return n * pivots


def pivot_tournament(
    candidates: list[Candidate], pivots: int = DEFAULT_PIVOTS
) -> Candidate:
    """Select the best candidate via pairwise Bradley-Terry wins against `pivots` pivots."""
    if not candidates:
        raise ValueError("no candidates to rank")
    if len(candidates) == 1:
        return candidates[0]

    ordered = sorted(candidates, key=lambda c: c.rating, reverse=True)
    pivot_set = ordered[: max(1, min(pivots, len(ordered)))]

    def expected_wins(c: Candidate) -> float:
        return sum(bradley_terry_prob(c.rating, p.rating) for p in pivot_set)

    return max(candidates, key=expected_wins)
