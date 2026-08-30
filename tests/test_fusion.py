"""Fusion tests.

The properties checked here are the ones the ablation depends on: that fusion
can rank something above what either retriever put first, that it is
deterministic, and that it records provenance.
"""

import pytest

from researchlens.retrieval.fusion import reciprocal_rank_fusion


def test_agreement_beats_a_single_first_place():
    """The property that makes hybrid retrieval worth having.

    "c" is second on both lists and first on neither. Two second places
    outscore one first place, so fusion surfaces the passage both retrievers
    liked over the one each liked alone.
    """
    fused = reciprocal_rank_fusion({"dense": ["a", "c"], "bm25": ["b", "c"]})
    assert fused[0][0] == "c"


def test_sources_record_who_proposed_what():
    fused = reciprocal_rank_fusion({"dense": ["a", "c"], "bm25": ["b", "c"]})
    by_id = {cid: srcs for cid, _, srcs in fused}
    assert by_id["c"] == frozenset({"dense", "bm25"})
    assert by_id["a"] == frozenset({"dense"})
    assert by_id["b"] == frozenset({"bm25"})


def test_scores_match_the_rrf_formula():
    k = 60
    fused = dict((cid, s) for cid, s, _ in reciprocal_rank_fusion({"dense": ["a", "b"]}, k=k))
    assert fused["a"] == pytest.approx(1 / (k + 1))
    assert fused["b"] == pytest.approx(1 / (k + 2))


def test_single_retriever_preserves_its_ordering():
    """Every ablation row runs through fusion, including the ones with one
    retriever, so fusion must be an identity on the ordering in that case."""
    order = ["x", "y", "z", "w"]
    fused = [cid for cid, _, _ in reciprocal_rank_fusion({"dense": order})]
    assert fused == order


def test_ties_break_deterministically():
    """Two runs over one index must produce identical metrics.

    Both ids sit at rank 1 of their own list and so score identically; without
    an explicit tiebreak, dict ordering would decide, and Recall@5 could differ
    between runs for no reason anyone could find.
    """
    a = reciprocal_rank_fusion({"dense": ["m"], "bm25": ["n"]})
    b = reciprocal_rank_fusion({"bm25": ["n"], "dense": ["m"]})
    assert [c for c, _, _ in a] == [c for c, _, _ in b] == ["m", "n"]


def test_weights_shift_the_ordering():
    even = reciprocal_rank_fusion({"dense": ["d"], "bm25": ["b"]})
    assert [c for c, _, _ in even] == ["b", "d"]  # tie, broken alphabetically
    tilted = reciprocal_rank_fusion({"dense": ["d"], "bm25": ["b"]}, weights={"dense": 2.0})
    assert tilted[0][0] == "d"
