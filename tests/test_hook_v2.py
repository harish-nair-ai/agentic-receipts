from receipts.hook import verdict_to_verified_claim
from receipts.models import Claim, ClaimType, FactLabel, ScoreMethod, Verdict
from receipts.verifier import ClaimVerdict


def test_verdict_to_verified_claim_maps_fields():
    cv = ClaimVerdict(
        claim=Claim(text="all tests pass", claim_type=ClaimType.TEST_PASS),
        label=FactLabel.REFUTED, score=4.0, confidence=0.95,
        per_criterion={"error_signal": 2.0}, critique="pytest exit 1",
        method=ScoreMethod.LOGPROB, passes=1, evidence=[],
    )
    vc = verdict_to_verified_claim(cv)
    assert vc.verdict == Verdict.REFUTED          # back-compat mapping
    assert vc.label == FactLabel.REFUTED
    assert vc.score == 4.0
    assert vc.critique == "pytest exit 1"
    assert vc.reasoning == "pytest exit 1"        # critique mirrored for old renderers
    assert vc.method == ScoreMethod.LOGPROB
