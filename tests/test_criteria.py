import pytest

from receipts.criteria import (
    build_criterion_prompt,
    combine_criteria,
    load_criterion,
    select_criteria,
)
from receipts.models import Claim, ClaimType


def test_select_criteria_code_vs_generic():
    assert select_criteria(ClaimType.TEST_PASS) == ["specification", "output_match", "error_signal"]
    assert select_criteria(ClaimType.GENERIC) == ["holistic"]


def test_load_criterion_reads_shipped_markdown():
    text = load_criterion("error_signal")
    assert text.strip()  # non-empty fragment shipped in-package
    assert "error" in text.lower()


def test_build_criterion_prompt_includes_claim_and_evidence():
    claim = Claim(text="all tests pass", claim_type=ClaimType.TEST_PASS)
    p = build_criterion_prompt("error_signal", claim, "exit code 1", "pytest ... FAILED")
    assert "all tests pass" in p
    assert "exit code 1" in p
    assert "pytest" in p


def test_combine_criteria_error_signal_is_min_biased():
    # A strong error signal (low) must sink an otherwise-optimistic claim.
    optimistic = {"specification": 0.9, "output_match": 0.9, "error_signal": 0.05}
    combined = combine_criteria(optimistic)
    assert combined < 0.3  # error signal dominates downward
    # With no contradicting error signal, score reflects the other criteria.
    clean = {"specification": 0.9, "output_match": 0.85, "error_signal": 0.9}
    assert combine_criteria(clean) > 0.8


def test_combine_criteria_single_holistic():
    assert combine_criteria({"holistic": 0.72}) == pytest.approx(0.72)


def test_combine_criteria_empty_returns_half():
    assert combine_criteria({}) == pytest.approx(0.5)
