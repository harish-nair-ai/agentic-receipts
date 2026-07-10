"""The verifier core: fast-path → calibrated scorer ensemble → uncertainty control.

Produces a three-way calibrated verdict (SUPPORTED / REFUTED / NOT_ENOUGH_INFO) per
claim with a 0-100 score, confidence, per-criterion breakdown, and a specific critique.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean, pstdev

from receipts.config import Config, check_independence
from receipts.criteria import build_criterion_prompt, combine_criteria, select_criteria
from receipts.evidence import match_evidence
from receipts.models import Claim, Evidence, FactLabel, ParsedTranscript, ScoreMethod
from receipts.scorer import ScorePass, generate_critique, score_pass

SUPPORTED_BAND = 85.0       # score ≥ this → SUPPORTED (when not abstaining)
REFUTED_BAND = 45.0         # score < this → REFUTED (when not abstaining)
ESCALATE_PEAKINESS = 0.75   # first-pass peakiness below this triggers adaptive-K escalation


@dataclass
class ClaimVerdict:
    """The verifier's decision for one claim."""

    claim: Claim
    label: FactLabel
    score: float                       # 0..100
    confidence: float                  # 0..1
    per_criterion: dict[str, float]    # criterion name → 0..100
    critique: str
    method: ScoreMethod
    passes: int
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class SessionReport:
    """All claim verdicts for a session plus checker metadata."""

    verdicts: list[ClaimVerdict]
    checker_model: str
    checker_independent: bool
    duration_ms: int


def label_for(score: float, confidence: float, min_confidence: float) -> FactLabel:
    """Apply score→label bands, with confidence-gated abstention (meta-honesty)."""
    if confidence < min_confidence:
        return FactLabel.NOT_ENOUGH_INFO
    if score >= SUPPORTED_BAND:
        return FactLabel.SUPPORTED
    if score < REFUTED_BAND:
        return FactLabel.REFUTED
    return FactLabel.NOT_ENOUGH_INFO


def aggregate_confidence(score01s: list[float], peakinesses: list[float]) -> float:
    """Confidence from cross-pass agreement × mean distribution peakiness.

    Agreement = 1 - min(1, 2·pstdev(per-pass scores)); with <2 passes agreement is 1.0
    and confidence is carried entirely by peakiness.
    """
    if not score01s:
        return 0.0
    agreement = 1.0 - min(1.0, 2.0 * pstdev(score01s)) if len(score01s) > 1 else 1.0
    mean_peak = mean(peakinesses) if peakinesses else 0.0
    return agreement * mean_peak


def fast_path(claim: Claim, evidence: list[Evidence]) -> ClaimVerdict | None:
    """Deterministic entailment: decide with certainty when hard evidence is conclusive."""
    if not evidence:
        return None
    # Hard contradiction (e.g. failing test exit code) → REFUTED with certainty.
    if all(not e.supports_claim for e in evidence):
        return ClaimVerdict(
            claim=claim, label=FactLabel.REFUTED, score=0.0, confidence=1.0,
            per_criterion={}, critique=evidence[0].content, method=ScoreMethod.DETERMINISTIC,
            passes=0, evidence=evidence,
        )
    # Hard support (e.g. matching file write) → SUPPORTED with certainty.
    if all(e.supports_claim for e in evidence):
        return ClaimVerdict(
            claim=claim, label=FactLabel.SUPPORTED, score=100.0, confidence=1.0,
            per_criterion={}, critique=evidence[0].content, method=ScoreMethod.DETERMINISTIC,
            passes=0, evidence=evidence,
        )
    return None  # mixed evidence → send to the LLM scorer


def _abstain(claim: Claim, evidence: list[Evidence], reason: str) -> ClaimVerdict:
    """Evidence-only / no-checker abstention (deepest fallback rung)."""
    return ClaimVerdict(
        claim=claim, label=FactLabel.NOT_ENOUGH_INFO, score=50.0, confidence=0.0,
        per_criterion={}, critique=reason, method=ScoreMethod.NONE, passes=0, evidence=evidence,
    )


def _run_pass(prompts: dict[str, str], config: Config) -> dict[str, ScorePass] | None:
    """Score every criterion once. Returns None if the scorer is entirely unavailable."""
    out: dict[str, ScorePass] = {}
    for name, prompt in prompts.items():
        sp = score_pass(prompt, config)
        if sp is not None:
            out[name] = sp
    return out or None


def verify_claim(
    claim: Claim,
    evidence: list[Evidence],
    transcript_context: str,
    config: Config,
    *,
    agent: str = "claude-code",
) -> ClaimVerdict:
    """Verify one claim: fast-path, else adaptive-K calibrated scoring + abstention."""
    fp = fast_path(claim, evidence)
    if fp is not None:
        return fp

    evidence_text = "\n".join(f"- {e.source.value}: {e.content}" for e in evidence)
    criteria = select_criteria(claim.claim_type)
    prompts = {c: build_criterion_prompt(c, claim, evidence_text, transcript_context)
               for c in criteria}

    # Pass 1.
    pass_criteria = _run_pass(prompts, config)
    if pass_criteria is None:
        return _abstain(claim, evidence, "No calibrated checker available for this claim.")

    per_pass_criteria: list[dict[str, ScorePass]] = [pass_criteria]
    method = next(iter(pass_criteria.values())).method

    # Adaptive-K escalation: low first-pass peakiness → run up to K_max passes.
    first_peak = mean(sp.peakiness for sp in pass_criteria.values())
    if first_peak < ESCALATE_PEAKINESS and config.score_passes > 1:
        for _ in range(config.score_passes - 1):
            extra = _run_pass(prompts, config)
            if extra is not None:
                per_pass_criteria.append(extra)

    # Combine: per criterion average across passes → weak-verifier combine → 0..100.
    per_criterion01: dict[str, float] = {}
    for name in criteria:
        vals = [p[name].score01 for p in per_pass_criteria if name in p]
        per_criterion01[name] = mean(vals) if vals else 0.5
    combined01 = combine_criteria(per_criterion01)
    score = combined01 * 100.0

    # Confidence: per-pass combined scores + mean peakiness.
    per_pass_combined = [
        combine_criteria({n: p[n].score01 for n in p}) for p in per_pass_criteria
    ]
    peaks = [sp.peakiness for p in per_pass_criteria for sp in p.values()]
    confidence = aggregate_confidence(per_pass_combined, peaks)

    label = label_for(score, confidence, config.min_confidence)
    critique = generate_critique(
        claim.text, evidence_text, transcript_context, label.value, config
    ) or "No specific critique available."

    return ClaimVerdict(
        claim=claim,
        label=label,
        score=score,
        confidence=confidence,
        per_criterion={n: v * 100.0 for n, v in per_criterion01.items()},
        critique=critique,
        method=method,
        passes=len(per_pass_criteria),
        evidence=evidence,
    )


def verify_session(
    claims: list[Claim],
    transcript: ParsedTranscript,
    transcript_context: str,
    config: Config,
    *,
    agent: str = "claude-code",
) -> SessionReport:
    """Verify every claim in a session and assemble the report."""
    start = time.time()
    independent = check_independence(config, agent)
    verdicts = [
        verify_claim(claim, match_evidence(claim, transcript), transcript_context,
                     config, agent=agent)
        for claim in claims
    ]
    return SessionReport(
        verdicts=verdicts,
        checker_model=config.checker_model,
        checker_independent=independent,
        duration_ms=int((time.time() - start) * 1000),
    )
