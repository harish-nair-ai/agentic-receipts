# tests/test_models_v2.py
from receipts.models import (
    Claim,
    ClaimType,
    FactLabel,
    Receipt,
    ScoreMethod,
    Verdict,
    VerifiedClaim,
    verdict_for_label,
)


def test_factlabel_and_scoremethod_values():
    assert FactLabel.SUPPORTED == "supported"
    assert FactLabel.REFUTED == "refuted"
    assert FactLabel.NOT_ENOUGH_INFO == "not_enough_info"
    assert ScoreMethod.LOGPROB == "logprob"
    assert ScoreMethod.SAMPLED == "sampled"


def test_verdict_for_label_maps_backcompat():
    assert verdict_for_label(FactLabel.SUPPORTED) == Verdict.VERIFIED
    assert verdict_for_label(FactLabel.REFUTED) == Verdict.REFUTED
    assert verdict_for_label(FactLabel.NOT_ENOUGH_INFO) == Verdict.UNVERIFIED


def test_verifiedclaim_new_fields_optional_and_backcompat():
    # Old-style construction (no new fields) still works.
    vc = VerifiedClaim(claim=Claim(text="x"), verdict=Verdict.VERIFIED)
    assert vc.label is None and vc.score is None and vc.per_criterion == {}
    # New-style construction populates the calibrated fields.
    vc2 = VerifiedClaim(
        claim=Claim(text="all tests pass", claim_type=ClaimType.TEST_PASS),
        verdict=Verdict.REFUTED,
        label=FactLabel.REFUTED,
        score=8.0,
        confidence=0.92,
        per_criterion={"error_signal": 5.0},
        critique="pytest exited 1 with 2 failures",
        method=ScoreMethod.LOGPROB,
        passes=1,
    )
    assert vc2.label == FactLabel.REFUTED
    assert vc2.per_criterion["error_signal"] == 5.0


def test_receipt_aggregates_over_calibrated_fields():
    claims = [
        VerifiedClaim(claim=Claim(text="a"), verdict=Verdict.VERIFIED,
                      label=FactLabel.SUPPORTED, score=95.0),
        VerifiedClaim(claim=Claim(text="b"), verdict=Verdict.REFUTED,
                      label=FactLabel.REFUTED, score=5.0),
        VerifiedClaim(claim=Claim(text="c"), verdict=Verdict.UNVERIFIED,
                      label=FactLabel.NOT_ENOUGH_INFO, score=60.0),
    ]
    r = Receipt(session_id="s1", claims=claims)
    assert r.supported_count == 1
    assert r.refuted_count == 1
    assert r.nei_count == 1
    # verified_done_score = mean of scores = (95+5+60)/3 ≈ 53.33
    assert round(r.verified_done_score, 1) == 53.3
    assert r.checker_independent is True  # default


def test_receipt_verified_done_score_empty_is_100():
    r = Receipt(session_id="s2", claims=[])
    assert r.verified_done_score == 100.0
