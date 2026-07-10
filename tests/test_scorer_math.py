import math

import pytest

from receipts.scorer import (
    SCORE_LETTERS,
    G,
    expected_score01,
    letter_value,
    peakiness,
    sampled_peakiness,
    sampled_score01,
)


def test_scale_constants():
    assert G == 20
    assert SCORE_LETTERS == "ABCDEFGHIJKLMNOPQRST"
    assert len(SCORE_LETTERS) == 20


def test_letter_value_endpoints_and_mid():
    assert letter_value("A") == 0.0
    assert letter_value("T") == 1.0
    assert letter_value("K") == pytest.approx((ord("K") - ord("A")) / 19)


def test_expected_score01_single_candidate():
    # Only "T" present → expectation is exactly value(T) = 1.0 after renormalization.
    assert expected_score01({"T": math.log(0.5)}) == pytest.approx(1.0)


def test_expected_score01_two_candidates_renormalized():
    # A (0.0) with p=0.25, T (1.0) with p=0.75 → renormalized expectation = 0.75.
    lp = {"A": math.log(0.25), "T": math.log(0.75)}
    assert expected_score01(lp) == pytest.approx(0.75)


def test_expected_score01_ignores_invalid_tokens():
    # "Z", " ", "hello" are not valid A..T letters and are dropped.
    lp = {"Z": math.log(0.5), "T": math.log(0.25), "A": math.log(0.25)}
    # Renormalize over A(0.25)->0.0 and T(0.25)->1.0 → 0.5
    assert expected_score01(lp) == pytest.approx(0.5)


def test_expected_score01_none_when_no_valid_candidate():
    assert expected_score01({"Z": math.log(0.9), "5": math.log(0.1)}) is None
    assert expected_score01({}) is None


def test_peakiness_concentrated_vs_flat():
    concentrated = peakiness({"T": math.log(0.95), "S": math.log(0.05)})
    flat = peakiness({"A": math.log(0.5), "T": math.log(0.5)})
    assert concentrated > 0.9
    assert flat == pytest.approx(0.5)


def test_sampled_score01_and_peakiness():
    assert sampled_score01(["T", "T", "T"]) == pytest.approx(1.0)
    assert sampled_score01([]) is None
    # All-same letters → maximum peakiness.
    assert sampled_peakiness(["K", "K", "K"]) == pytest.approx(1.0)
    # Spread letters → lower peakiness.
    assert sampled_peakiness(["A", "T"]) < 0.5
