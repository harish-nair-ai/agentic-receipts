from unittest.mock import patch

from receipts.config import Config, JudgeProvider
from receipts.models import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceSource,
    FactLabel,
    ScoreMethod,
)
from receipts.scorer import ScorePass
from receipts.verifier import (
    aggregate_confidence,
    fast_path,
    label_for,
    verify_claim,
)


def _cfg(**kw):
    return Config(provider=JudgeProvider.OPENAI, model="m", api_key="k", **kw)


def _sp(score01, peak=0.95):
    return ScorePass(score01=score01, peakiness=peak, method=ScoreMethod.LOGPROB, letter="T")


def test_fast_path_refutes_failing_tests_without_llm():
    claim = Claim(text="all tests pass", claim_type=ClaimType.TEST_PASS)
    ev = [Evidence(source=EvidenceSource.EXIT_CODE, content="exit 1", supports_claim=False)]
    v = fast_path(claim, ev)
    assert v is not None
    assert v.label == FactLabel.REFUTED
    assert v.score == 0.0
    assert v.method == ScoreMethod.DETERMINISTIC


def test_fast_path_supports_created_file_without_llm():
    claim = Claim(text="created src/utils.py", claim_type=ClaimType.FILE_CREATED)
    ev = [Evidence(source=EvidenceSource.FILE_WRITE, content="wrote src/utils.py",
                   supports_claim=True)]
    v = fast_path(claim, ev)
    assert v is not None and v.label == FactLabel.SUPPORTED and v.score == 100.0


def test_fast_path_returns_none_for_fuzzy_claim():
    claim = Claim(text="improved error handling", claim_type=ClaimType.GENERIC)
    assert fast_path(claim, []) is None


def test_label_for_bands_and_abstention():
    assert label_for(90.0, confidence=0.9, min_confidence=0.6) == FactLabel.SUPPORTED
    assert label_for(10.0, confidence=0.9, min_confidence=0.6) == FactLabel.REFUTED
    assert label_for(65.0, confidence=0.9, min_confidence=0.6) == FactLabel.NOT_ENOUGH_INFO
    # Low confidence forces abstention even on a decisive score.
    assert label_for(95.0, confidence=0.3, min_confidence=0.6) == FactLabel.NOT_ENOUGH_INFO


def test_aggregate_confidence_agreement_and_peakiness():
    # Perfect agreement + high peakiness → high confidence.
    assert aggregate_confidence([0.8, 0.8], [0.9, 0.9]) > 0.85
    # Disagreement collapses confidence.
    assert aggregate_confidence([0.1, 0.9], [0.9, 0.9]) < 0.5


def test_verify_claim_high_supported_single_pass():
    claim = Claim(text="improved logging", claim_type=ClaimType.GENERIC)
    with patch("receipts.verifier.score_pass", return_value=_sp(0.95, 0.95)) as m, \
         patch("receipts.verifier.generate_critique", return_value="log calls added"):
        v = verify_claim(claim, [], "ctx", _cfg(score_passes=3))
    assert v.label == FactLabel.SUPPORTED
    assert v.passes == 1  # clear + peaked → no escalation
    m.assert_called_once()


def test_verify_claim_escalates_on_low_peakiness():
    claim = Claim(text="refactored the parser", claim_type=ClaimType.REFACTORED)
    # First pass low peakiness → escalate; code claim has 3 criteria × passes calls.
    with patch("receipts.verifier.score_pass", return_value=_sp(0.6, 0.4)), \
         patch("receipts.verifier.generate_critique", return_value="unclear"):
        v = verify_claim(claim, [], "ctx", _cfg(score_passes=3))
    assert v.passes == 3  # escalated to K_max


def test_verify_claim_abstains_when_scorer_unavailable():
    claim = Claim(text="cleaned up things", claim_type=ClaimType.GENERIC)
    with patch("receipts.verifier.score_pass", return_value=None):
        v = verify_claim(claim, [], "ctx", _cfg())
    assert v.label == FactLabel.NOT_ENOUGH_INFO
    assert v.method == ScoreMethod.NONE
