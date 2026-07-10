from unittest.mock import patch

from receipts.config import Config, JudgeProvider
from receipts.hook import maybe_propose_fixes
from receipts.models import Claim, ClaimType, FactLabel, ScoreMethod, Verdict, VerifiedClaim
from receipts.retry import FixProposal


def _cfg(**kw):
    return Config(provider=JudgeProvider.OPENAI, model="m", api_key="k", **kw)


def _refuted_vc():
    return VerifiedClaim(
        claim=Claim(text="all tests pass", claim_type=ClaimType.TEST_PASS),
        verdict=Verdict.REFUTED, label=FactLabel.REFUTED, score=5.0,
        critique="pytest exit 1", method=ScoreMethod.LOGPROB,
    )


def test_maybe_propose_fixes_attaches_diff_but_does_not_apply_by_default():
    vcs = [_refuted_vc()]
    fix = FixProposal(diff="--- a/x\n+++ b/x\n", rating=0.9, explanation="fix it")
    with patch("receipts.hook.propose_fix", return_value=fix), \
         patch("receipts.hook.apply_fix") as apply_mock:
        maybe_propose_fixes(vcs, "ctx", _cfg(autofix=False))
    assert vcs[0].proposed_fix.startswith("--- a/x")
    assert vcs[0].fix_applied is False
    apply_mock.assert_not_called()


def test_maybe_propose_fixes_applies_when_autofix_enabled():
    vcs = [_refuted_vc()]
    fix = FixProposal(diff="--- a/x\n+++ b/x\n", rating=0.9, explanation="fix it")
    with patch("receipts.hook.propose_fix", return_value=fix), \
         patch("receipts.hook.apply_fix", return_value=True) as apply_mock:
        maybe_propose_fixes(vcs, "ctx", _cfg(autofix=True))
    assert vcs[0].fix_applied is True
    apply_mock.assert_called_once()


def test_maybe_propose_fixes_skips_non_refuted():
    vc = VerifiedClaim(claim=Claim(text="a"), verdict=Verdict.VERIFIED,
                       label=FactLabel.SUPPORTED, score=95.0)
    with patch("receipts.hook.propose_fix") as pf:
        maybe_propose_fixes([vc], "ctx", _cfg(autofix=True))
    pf.assert_not_called()
