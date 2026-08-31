"""Reranker tests, against a stub encoder.

Loading the real cross-encoder would test HuggingFace's weights rather than
this module. What can be wrong here is the pairing, the ordering, the
truncation and the tie-break.
"""

from researchlens.retrieval.rerank import CrossEncoderReranker
from researchlens.types import Chunk


class StubEncoder:
    """Scores by how many query words a passage contains."""

    def rerank(self, query, texts):
        words = set(query.lower().split())
        for t in texts:
            yield float(len(words & set(t.lower().split())))


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc:{i}", doc_id="doc", ordinal=i, text=text,
        section_kind="results", section_heading="Results",
        page_start=1, page_end=1, doc_title="A Paper",
    )


def _reranker():
    r = CrossEncoderReranker()
    r._encoder = StubEncoder()
    return r


def test_candidates_are_reordered_by_score():
    r = _reranker()
    out = r.rerank("kidney dice score", [
        _chunk(0, "unrelated text about optimisers"),
        _chunk(1, "kidney dice score reported here"),
    ], top_k=2)
    assert out[0][0] == "doc:1"
    assert out[0][1] > out[1][1]


def test_scores_stay_paired_with_their_own_chunk():
    """The failure this guards against is silent: a zip that drifts by one
    returns plausible passages with the wrong scores, and nothing looks wrong."""
    r = _reranker()
    out = dict(r.rerank("alpha beta", [
        _chunk(0, "alpha"), _chunk(1, "alpha beta"), _chunk(2, "gamma"),
    ], top_k=3))
    assert out["doc:1"] == 2.0
    assert out["doc:0"] == 1.0
    assert out["doc:2"] == 0.0


def test_output_is_truncated_to_top_k():
    r = _reranker()
    out = r.rerank("x", [_chunk(i, f"x {i}") for i in range(10)], top_k=3)
    assert len(out) == 3


def test_no_candidates_returns_nothing_and_loads_no_model():
    r = CrossEncoderReranker()  # deliberately no stub attached
    assert r.rerank("anything", [], top_k=5) == []


def test_ties_break_deterministically():
    r = _reranker()
    a = r.rerank("q", [_chunk(0, "none"), _chunk(1, "none")], top_k=2)
    b = r.rerank("q", [_chunk(1, "none"), _chunk(0, "none")], top_k=2)
    assert [c for c, _ in a] == [c for c, _ in b]
