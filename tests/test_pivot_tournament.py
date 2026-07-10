import pytest

from receipts.pivot_tournament import (
    DEFAULT_PIVOTS,
    Candidate,
    bradley_terry_prob,
    comparison_count,
    pivot_tournament,
)


def test_bradley_terry_symmetry_and_midpoint():
    assert bradley_terry_prob(1.0, 1.0) == pytest.approx(0.5)
    assert bradley_terry_prob(2.0, 0.0) > 0.8
    assert bradley_terry_prob(0.0, 2.0) < 0.2


def test_default_pivots_is_two():
    assert DEFAULT_PIVOTS == 2


def test_pivot_tournament_selects_highest_rated():
    cands = [Candidate(id="a", rating=0.2), Candidate(id="b", rating=0.9),
             Candidate(id="c", rating=0.5)]
    assert pivot_tournament(cands).id == "b"


def test_pivot_tournament_single_candidate():
    only = Candidate(id="solo", rating=0.3)
    assert pivot_tournament([only]).id == "solo"


def test_comparison_count_is_linear_Nk():  # noqa: N802
    # O(N·k): each of N candidates compared against `pivots` pivots.
    assert comparison_count(10, 2) == 20
    assert comparison_count(5, 2) == 10
