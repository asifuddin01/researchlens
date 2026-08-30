"""Dense retriever tests.

These use a stub encoder rather than the real ONNX model. Loading BGE would
make the suite slow and would test HuggingFace's weights rather than this
module's logic — the things that can actually be wrong here are the
normalisation, the top-k selection, the ordering and the cache key.
"""

import numpy as np
import pytest

from researchlens.retrieval.dense import DenseRetriever, _l2_normalise
from researchlens.types import Chunk


class StubEncoder:
    """Maps text to a fixed vector by keyword, so similarity is predictable."""

    VECTORS = {
        "kidney": [1.0, 0.0, 0.0],
        "brain": [0.0, 1.0, 0.0],
        "liver": [0.0, 0.0, 1.0],
    }

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        for t in texts:
            for word, vec in self.VECTORS.items():
                if word in t.lower():
                    yield np.array(vec, dtype=np.float32)
                    break
            else:
                yield np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc:{i}", doc_id="doc", ordinal=i, text=text,
        section_kind="results", section_heading="Results",
        page_start=1, page_end=1, doc_title="A Paper",
    )


@pytest.fixture
def retriever(tmp_path):
    r = DenseRetriever(cache_dir=tmp_path)
    r._encoder = StubEncoder()
    r.index([
        _chunk(0, "kidney segmentation performance"),
        _chunk(1, "brain tumour analysis"),
        _chunk(2, "liver volume estimation"),
    ])
    return r


def test_search_returns_the_nearest_chunk_first(retriever):
    hits = retriever.search("kidney", k=3)
    assert hits[0][0] == "doc:0"
    assert hits[0][1] == pytest.approx(1.0)


def test_results_are_ordered_by_descending_score(retriever):
    scores = [s for _, s in retriever.search("kidney", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_k_larger_than_the_corpus_is_clamped(retriever):
    # argpartition raises if k exceeds the array length; clamping must happen
    # before it, not be caught after.
    assert len(retriever.search("kidney", k=99)) == 3


def test_k_of_zero_returns_nothing(retriever):
    assert retriever.search("kidney", k=0) == []


def test_search_before_index_is_an_error(tmp_path):
    r = DenseRetriever(cache_dir=tmp_path)
    r._encoder = StubEncoder()
    with pytest.raises(RuntimeError, match="before index"):
        r.search("kidney", k=1)


def test_index_rejects_an_empty_corpus(tmp_path):
    r = DenseRetriever(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="nothing to index"):
        r.index([])


def test_embeddings_are_cached_and_reused(tmp_path):
    chunks = [_chunk(0, "kidney"), _chunk(1, "brain")]

    first = DenseRetriever(cache_dir=tmp_path)
    first._encoder = StubEncoder()
    first.index(chunks)
    assert first._encoder.calls > 0

    second = DenseRetriever(cache_dir=tmp_path)
    second._encoder = StubEncoder()
    second.index(chunks)
    # Loaded from disk, so the encoder was never asked to embed.
    assert second._encoder.calls == 0
    assert np.allclose(first._matrix, second._matrix)


def test_cache_key_changes_when_text_changes(tmp_path):
    """Ids alone would collide when the chunker is retuned to the same count,
    and a stale cache scoring against the wrong passages is invisible."""
    r = DenseRetriever(cache_dir=tmp_path)
    a = r._cache_key([_chunk(0, "kidney")])
    b = r._cache_key([_chunk(0, "brain")])
    assert a != b


def test_cache_key_changes_with_the_query_instruction(tmp_path):
    plain = DenseRetriever(cache_dir=tmp_path)
    instructed = DenseRetriever(cache_dir=tmp_path, query_instruction="Query: ")
    chunks = [_chunk(0, "kidney")]
    assert plain._cache_key(chunks) != instructed._cache_key(chunks)


def test_normalise_leaves_zero_vectors_finite():
    # A NaN here would propagate into every score and present as uniformly
    # broken retrieval rather than as one bad chunk.
    out = _l2_normalise(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))
    assert np.isfinite(out).all()
    assert out[1] == pytest.approx([0.6, 0.8])
