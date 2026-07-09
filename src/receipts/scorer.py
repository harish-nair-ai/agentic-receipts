"""Calibrated scoring — logprob expectation over an A-T 20-level ordinal scale.

Implements the base paper's Eq 3.1 (mirrors fine_grained_reward.py::extract_score):

    score01 = Σ_i value(tok_i) · p_i / Σ_i p_i      # renormalized over valid candidates
    value(letter) = (ord(letter) − ord('A')) / 19   # A→0.0 … T→1.0 ; p_i = exp(logprob_i)
"""

from __future__ import annotations

import math
from statistics import pstdev

G = 20
SCORE_LETTERS = "ABCDEFGHIJKLMNOPQRST"  # 20 ordinal levels, A = fully refuted, T = fully supported
_VALID = set(SCORE_LETTERS)


def _clean_letter(token: str) -> str | None:
    """Normalize a returned token to a single valid scoring letter, or None."""
    t = token.strip().upper()
    return t if len(t) == 1 and t in _VALID else None


def letter_value(letter: str) -> float:
    """Map a scoring letter to its ordinal value in [0, 1]."""
    t = _clean_letter(letter)
    if t is None:
        raise ValueError(f"not a valid scoring letter: {letter!r}")
    return (ord(t) - ord("A")) / (G - 1)


def _valid_probs(logprobs: dict[str, float]) -> dict[str, float]:
    """Collapse raw {token: logprob} to {letter: probability} over valid letters."""
    probs: dict[str, float] = {}
    for token, lp in logprobs.items():
        letter = _clean_letter(token)
        if letter is not None:
            probs[letter] = probs.get(letter, 0.0) + math.exp(lp)
    return probs


def expected_score01(logprobs: dict[str, float]) -> float | None:
    """Probability-weighted expectation (Eq 3.1), renormalized over valid candidates."""
    probs = _valid_probs(logprobs)
    denom = sum(probs.values())
    if denom == 0.0:
        return None
    return sum(letter_value(letter) * p for letter, p in probs.items()) / denom


def peakiness(logprobs: dict[str, float]) -> float:
    """Renormalized probability mass on the modal valid letter (0..1). 0.0 if none valid."""
    probs = _valid_probs(logprobs)
    denom = sum(probs.values())
    if denom == 0.0:
        return 0.0
    return max(probs.values()) / denom


def sampled_score01(letters: list[str]) -> float | None:
    """Mean ordinal value over valid sampled letters, or None if none valid."""
    values = [letter_value(c) for c in letters if _clean_letter(c) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def sampled_peakiness(letters: list[str]) -> float:
    """Concentration of sampled letters: 1 - min(1, 2·pstdev(values)). 1.0 for <2 samples."""
    values = [letter_value(c) for c in letters if _clean_letter(c) is not None]
    if len(values) < 2:
        return 1.0
    return 1.0 - min(1.0, 2.0 * pstdev(values))
