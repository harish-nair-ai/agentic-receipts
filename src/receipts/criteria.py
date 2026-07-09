"""Criteria decomposition — a weak-verifier ensemble over a claim (2506.18203).

Code claims are scored on three sub-criteria (the base paper's coding triad);
non-code/generic claims use a single holistic criterion. `error_signal` is
min-biased in combination so one strong contradiction sinks an optimistic claim.
"""

from __future__ import annotations

import math
from functools import cache
from importlib import resources

from receipts.models import Claim, ClaimType

CODE_CLAIM_TYPES: set[ClaimType] = {
    ClaimType.TEST_PASS,
    ClaimType.BUG_FIXED,
    ClaimType.FEATURE_ADDED,
    ClaimType.REFACTORED,
    ClaimType.FILE_MODIFIED,
    ClaimType.COMMAND_RUN,
}

_CODE_CRITERIA = ["specification", "output_match", "error_signal"]
_GENERIC_CRITERIA = ["holistic"]


def select_criteria(claim_type: ClaimType) -> list[str]:
    """Pick the criterion set for a claim type."""
    return list(_CODE_CRITERIA) if claim_type in CODE_CLAIM_TYPES else list(_GENERIC_CRITERIA)


@cache
def load_criterion(name: str) -> str:
    """Load a shipped criterion prompt fragment by name."""
    return (resources.files("receipts") / "criteria" / f"{name}.md").read_text(encoding="utf-8")


def build_criterion_prompt(
    name: str, claim: Claim, evidence_text: str, transcript_context: str
) -> str:
    """Assemble the scoring prompt body for one criterion."""
    return (
        f"{load_criterion(name)}\n\n"
        f"CLAIM: {claim.text}\n\n"
        f"DETERMINISTIC EVIDENCE:\n{evidence_text or 'None found.'}\n\n"
        f"TRANSCRIPT CONTEXT:\n{transcript_context[:3000]}"
    )


def combine_criteria(scores: dict[str, float]) -> float:
    """Combine per-criterion 0-1 scores into one 0-1 score.

    error_signal is min-biased: the combined score is the geometric mean of the other
    criteria's average and the error_signal score, so one strong contradiction
    dominates downward (geometric mean is pulled toward the smaller factor). Without an
    error_signal criterion the plain mean is used. Empty → 0.5.
    """
    if not scores:
        return 0.5
    if "error_signal" in scores:
        others = [v for k, v in scores.items() if k != "error_signal"]
        mean_others = sum(others) / len(others) if others else scores["error_signal"]
        return math.sqrt(max(0.0, mean_others) * max(0.0, scores["error_signal"]))
    return sum(scores.values()) / len(scores)
