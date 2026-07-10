import io

from receipts.models import (
    Claim,
    ClaimType,
    FactLabel,
    Receipt,
    ScoreMethod,
    Verdict,
    VerifiedClaim,
)
from receipts.render import render_receipt


def _receipt(**kw):
    claims = [
        VerifiedClaim(
            claim=Claim(text="all tests pass", claim_type=ClaimType.TEST_PASS),
            verdict=Verdict.REFUTED, label=FactLabel.REFUTED, score=4.0, confidence=0.95,
            per_criterion={"error_signal": 2.0}, critique="pytest exit 1, 2 failures",
            method=ScoreMethod.LOGPROB, passes=1,
        ),
        VerifiedClaim(
            claim=Claim(text="improved error handling", claim_type=ClaimType.GENERIC),
            verdict=Verdict.UNVERIFIED, label=FactLabel.NOT_ENOUGH_INFO, score=60.0,
            confidence=0.4, critique="no try/except changes found", method=ScoreMethod.SAMPLED,
            passes=3,
        ),
    ]
    return Receipt(session_id="s1", claims=claims, judge_model="gemini-3-flash", **kw)


def test_card_shows_verified_done_score_and_counts():
    buf = io.StringIO()
    render_receipt(_receipt(), file=buf)
    out = buf.getvalue()
    assert "Verified-Done Score" in out
    assert "/100" in out
    assert "pytest exit 1" in out                 # critique rendered
    assert "improved error handling" in out


def test_card_shows_method_tags():
    buf = io.StringIO()
    render_receipt(_receipt(), file=buf)
    out = buf.getvalue().lower()
    assert "calibrated" in out or "logprob" in out
    assert "sampled" in out


def test_card_warns_when_not_independent():
    buf = io.StringIO()
    render_receipt(_receipt(checker_independent=False), file=buf)
    assert "independ" in buf.getvalue().lower()  # maker≠checker warning surfaced
