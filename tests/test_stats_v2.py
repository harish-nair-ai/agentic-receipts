import io

from receipts.models import (
    Claim,
    FactLabel,
    Receipt,
    Verdict,
    VerifiedClaim,
)
from receipts.stats import render_stats


def _receipt():
    return Receipt(
        session_id="s1",
        claims=[
            VerifiedClaim(claim=Claim(text="a"), verdict=Verdict.VERIFIED,
                          label=FactLabel.SUPPORTED, score=95.0),
            VerifiedClaim(claim=Claim(text="b"), verdict=Verdict.REFUTED,
                          label=FactLabel.REFUTED, score=5.0),
            VerifiedClaim(claim=Claim(text="c"), verdict=Verdict.UNVERIFIED,
                          label=FactLabel.NOT_ENOUGH_INFO, score=60.0),
        ],
    )


def test_stats_reports_calibrated_aggregates():
    buf = io.StringIO()
    render_stats([_receipt()], days=7, file=buf)
    out = buf.getvalue()
    assert "Verified-Done Score" in out
    assert "refuted" in out.lower()
    assert "not-enough-info" in out.lower() or "abstain" in out.lower()


def test_stats_handles_v1_records_without_scores():
    v1 = Receipt(
        session_id="old",
        claims=[VerifiedClaim(claim=Claim(text="x"), verdict=Verdict.VERIFIED)],
    )
    buf = io.StringIO()
    render_stats([v1], days=7, file=buf)  # must not raise
    assert "session" in buf.getvalue().lower()
