"""Metric tests against hand-computed values.

Every expected number here was worked out by hand rather than recorded from a
run, because a test that captures whatever the code currently does cannot
detect that the code is wrong.
"""

import math

import pytest

from eval.metrics import (
    aggregate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Ranked results, relevant = {b, e}. Positions (1-indexed): b=2, e=5.
R = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
REL = {"b", "e"}


def test_recall_counts_labels_found_not_positions():
    assert recall_at_k(R, REL, 1) == 0.0          # a only
    assert recall_at_k(R, REL, 2) == 0.5          # b
    assert recall_at_k(R, REL, 5) == 1.0          # b and e
    assert recall_at_k(R, REL, 10) == 1.0         # cannot exceed 1


def test_recall_is_zero_when_nothing_is_labelled():
    # Guards against a ZeroDivisionError on a malformed ground-truth row.
    assert recall_at_k(R, set(), 5) == 0.0


def test_precision_divides_by_k_not_by_hits():
    assert precision_at_k(R, REL, 5) == pytest.approx(2 / 5)
    assert precision_at_k(R, REL, 2) == pytest.approx(1 / 2)
    assert precision_at_k(R, REL, 0) == 0.0


def test_reciprocal_rank_uses_the_first_hit_only():
    assert reciprocal_rank(R, REL) == pytest.approx(1 / 2)      # b at rank 2
    assert reciprocal_rank(R, {"e"}) == pytest.approx(1 / 5)
    assert reciprocal_rank(R, {"zzz"}) == 0.0


def test_reciprocal_rank_respects_the_cutoff():
    # e is at rank 5, so a cutoff of 4 must not find it.
    assert reciprocal_rank(R, {"e"}, k=4) == 0.0
    assert reciprocal_rank(R, {"e"}, k=5) == pytest.approx(1 / 5)


def test_ndcg_matches_hand_computation():
    # DCG  = 1/log2(3) + 1/log2(6)   (ranks 2 and 5)
    # IDCG = 1/log2(2) + 1/log2(3)   (ranks 1 and 2)
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(R, REL, 10) == pytest.approx(dcg / idcg)


def test_ndcg_is_one_for_perfect_ordering():
    assert ndcg_at_k(["b", "e", "a"], REL, 10) == pytest.approx(1.0)


def test_ndcg_rewards_order_where_recall_cannot_see_it():
    """The property that justifies reporting nDCG alongside recall.

    Both orderings retrieve both labelled passages inside k, so recall is
    identical. Only nDCG can tell that one put them first — which is exactly
    the improvement a reranker is supposed to make.
    """
    good = ["b", "e", "x", "y", "z"]
    poor = ["x", "y", "z", "b", "e"]
    assert recall_at_k(good, REL, 5) == recall_at_k(poor, REL, 5) == 1.0
    assert ndcg_at_k(good, REL, 5) > ndcg_at_k(poor, REL, 5)


def test_aggregate_macro_averages_over_questions():
    """One question with two labels must not outweigh one with a single label."""
    per_q = [
        (["b", "z"], {"b"}),        # recall@5 = 1.0
        (["x", "y"], {"q", "r"}),   # recall@5 = 0.0
    ]
    score = aggregate("test", per_q, [10.0, 20.0, 30.0])
    assert score.n_questions == 2
    assert score.recall_at_5 == pytest.approx(0.5)
    assert score.median_latency_ms == 20.0


def test_aggregate_refuses_an_empty_question_set():
    # A silent 0.000 across the table would look like a catastrophic
    # regression rather than a file that failed to load.
    with pytest.raises(ValueError, match="no questions"):
        aggregate("test", [], [])
