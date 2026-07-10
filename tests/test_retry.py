from unittest.mock import patch

from receipts.config import Config, JudgeProvider
from receipts.models import Claim, ClaimType, FactLabel, ScoreMethod
from receipts.retry import (
    DEFAULT_CANDIDATES,
    FixProposal,
    propose_fix,
    select_fix,
)
from receipts.verifier import ClaimVerdict


def _cfg(**kw):
    return Config(provider=JudgeProvider.OPENAI, model="m", api_key="k", **kw)


def _refuted():
    return ClaimVerdict(
        claim=Claim(text="all tests pass", claim_type=ClaimType.TEST_PASS),
        label=FactLabel.REFUTED, score=5.0, confidence=0.9, per_criterion={},
        critique="pytest exit 1", method=ScoreMethod.LOGPROB, passes=1, evidence=[],
    )


def test_select_fix_picks_highest_rated():
    fixes = [
        FixProposal(diff="diff-a", rating=0.2, explanation="a"),
        FixProposal(diff="diff-b", rating=0.9, explanation="b"),
    ]
    with patch("receipts.retry.score_fix", side_effect=lambda f, *a, **k: f.rating):
        best = select_fix(fixes, _cfg())
    assert best.diff == "diff-b"


def test_select_fix_empty_returns_none():
    assert select_fix([], _cfg()) is None


def test_propose_fix_returns_winning_proposal():
    fixes = [FixProposal(diff="d1", rating=0.0, explanation="x"),
             FixProposal(diff="d2", rating=0.0, explanation="y")]
    with patch("receipts.retry.generate_fixes", return_value=fixes), \
         patch("receipts.retry.score_fix", side_effect=[0.3, 0.8]):
        out = propose_fix(_refuted(), "ctx", _cfg())
    assert out.diff == "d2"


def test_default_candidates_is_three():
    assert DEFAULT_CANDIDATES == 3
