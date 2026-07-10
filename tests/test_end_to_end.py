from pathlib import Path
from unittest.mock import patch

from receipts.claims import extract_claims
from receipts.config import Config, JudgeProvider
from receipts.models import ClaimType, FactLabel, ScoreMethod
from receipts.scorer import ScorePass
from receipts.transcript import parse_transcript
from receipts.verifier import verify_session

FIXTURE = Path(__file__).parent / "fixtures" / "session_refuted_and_ambiguous.jsonl"


def _cfg():
    return Config(provider=JudgeProvider.OPENAI, model="gpt-x", api_key="k", min_confidence=0.6)


def test_end_to_end_refuted_and_abstained():
    transcript = parse_transcript(FIXTURE)
    claims = extract_claims(transcript.final_message)
    # The ambiguous "improved error handling" claim scores mid with low peakiness → NEI.
    low = ScorePass(score01=0.6, peakiness=0.3, method=ScoreMethod.LOGPROB, letter="M")
    with patch("receipts.verifier.score_pass", return_value=low), \
         patch("receipts.verifier.generate_critique", return_value="no try/except changes found"):
        report = verify_session(claims, transcript,
                                "pytest ... exit 1 ... 2 failed", _cfg())
    # "all tests pass" is refuted deterministically by the pytest exit-1 fast-path.
    test_claim = next(v for v in report.verdicts if v.claim.claim_type == ClaimType.TEST_PASS)
    assert test_claim.label == FactLabel.REFUTED
    assert test_claim.method == ScoreMethod.DETERMINISTIC
    # The fuzzy claim abstains rather than guessing.
    fuzzy = [v for v in report.verdicts if v.claim.claim_type != ClaimType.TEST_PASS]
    assert fuzzy and all(v.label == FactLabel.NOT_ENOUGH_INFO for v in fuzzy)
