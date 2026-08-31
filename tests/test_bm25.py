"""BM25 tests.

The tokeniser gets the most attention, because it is where lexical retrieval
either keeps or destroys the identifiers it exists to find.
"""

import pytest

from researchlens.retrieval.bm25 import BM25Retriever, tokenise
from researchlens.types import Chunk


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc:{i}", doc_id="doc", ordinal=i, text=text,
        section_kind="results", section_heading="Results",
        page_start=1, page_end=1, doc_title="A Paper",
    )


def test_identifiers_survive_tokenisation():
    """The reason the lexical half exists. Splitting any of these turns an
    exact match into a topical one."""
    got = tokenise("U-Net reached 0.94 Dice on KiTS23 using AdamW and scGPT")
    for want in ("u-net", "0.94", "kits23", "adamw", "scgpt"):
        assert want in got, f"{want} lost: {got}"


def test_stopwords_are_removed_but_negation_is_not():
    got = tokenise("the model was not evaluated on all of the data")
    assert "the" not in got and "was" not in got
    # "not" changes the meaning of a question about negative results, and
    # "all" carries the title of a well-known paper.
    assert "not" in got and "all" in got


def test_exact_term_beats_topical_prose():
    r = BM25Retriever()
    r.index([
        _chunk(0, "The network was trained with the AdamW optimizer at 1e-4."),
        _chunk(1, "We discuss optimisation strategies for deep neural networks at length."),
    ])
    assert r.search("AdamW", k=2)[0][0] == "doc:0"


def test_a_term_in_no_document_scores_nothing():
    r = BM25Retriever()
    r.index([_chunk(0, "kidney segmentation results")])
    assert r.search("xylophone", k=5) == []


def test_saturation_stops_repetition_from_winning():
    """A passage repeating a term twenty times should not beat one that uses it
    twice and actually answers the question."""
    r = BM25Retriever()
    r.index([
        _chunk(0, "perturbation " * 20),
        _chunk(1, "The perturbation was applied to each cell and the perturbation "
                  "effect was measured by comparing expression profiles before and after."),
    ])
    hits = dict(r.search("perturbation", k=2))
    # Both match; the point is that repetition alone does not run away with it.
    assert hits["doc:0"] < hits["doc:1"] * 3


def test_results_are_ordered_and_clamped():
    r = BM25Retriever()
    r.index([_chunk(i, f"segmentation result {i}") for i in range(5)])
    hits = r.search("segmentation", k=3)
    assert len(hits) == 3
    assert [s for _, s in hits] == sorted([s for _, s in hits], reverse=True)


def test_ties_break_deterministically():
    """Two runs over one index must produce identical metrics."""
    docs = [_chunk(0, "identical text here"), _chunk(1, "identical text here")]
    a = BM25Retriever(); a.index(docs)
    b = BM25Retriever(); b.index(list(reversed(docs)))
    assert [c for c, _ in a.search("identical", k=2)] == [c for c, _ in b.search("identical", k=2)]


def test_search_before_index_is_an_error():
    with pytest.raises(RuntimeError, match="before index"):
        BM25Retriever().search("x", k=1)


def test_index_rejects_an_empty_corpus():
    with pytest.raises(ValueError, match="nothing to index"):
        BM25Retriever().index([])


def test_a_term_in_every_document_does_not_push_results_down():
    """Unfloored idf goes negative for very common terms, which would make a
    passage score worse for containing the query word."""
    r = BM25Retriever()
    r.index([_chunk(i, "cells were measured") for i in range(4)])
    assert all(s >= 0 for _, s in r.search("cells measured", k=4))
