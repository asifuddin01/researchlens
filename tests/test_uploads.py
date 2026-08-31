"""Uploaded papers: isolation, guards and eviction.

The test that matters most here is the boring one — that a paper uploaded in
one session never appears in another's results. The Space is a single process
serving every visitor, so the failure this prevents is a stranger's manuscript
quoted back, with a citation, at someone who never uploaded it.
"""

from __future__ import annotations

import numpy as np
import pytest

from researchlens.retrieval.dense import DenseRetriever
from researchlens.retrieval.pipeline import RetrievalConfig
from researchlens.uploads import UploadError, UploadStore
from tests.pdfs import make_pdf

HYBRID = RetrievalConfig("test", use_dense=True, use_bm25=True, use_rerank=False,
                         candidates=20, top_k=8)


class StubEncoder:
    """One dimension per keyword, so similarity is decidable by eye."""

    WORDS = ("kidney", "widget", "perturbation")

    def embed(self, texts, **kwargs):
        for t in texts:
            low = t.lower()
            v = np.array([1.0 if w in low else 0.0 for w in self.WORDS], dtype=np.float32)
            if not v.any():
                v = np.array([0.1, 0.1, 0.1], dtype=np.float32)
            yield v


class StubDense(DenseRetriever):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._encoder = StubEncoder()


@pytest.fixture(autouse=True)
def _no_real_weights(monkeypatch):
    """Never load 67 MB of ONNX to test bookkeeping."""
    monkeypatch.setattr("researchlens.uploads.DenseRetriever", StubDense)


def paper(topic: str, title: str) -> bytes:
    body = [f"This paper reports that {topic} segmentation improves by four points."] * 10
    return make_pdf([title, "Abstract"] + body, [16.0, 13.0] + [11.0] * len(body))


def store(**kw) -> UploadStore:
    return UploadStore(embedding_model="stub", reranker=None, **kw)


# ---- isolation -------------------------------------------------------------

def test_one_session_never_sees_anothers_paper():
    s = store()
    s.add("alice", paper("kidney", "Kidney Segmentation At Scale"), "alice.pdf")
    s.add("bob", paper("widget", "Widget Calibration In Practice"), "bob.pdf")

    for who, expect in (("alice", "kidney"), ("bob", "widget")):
        hits = s.search(who, f"{expect} results", HYBRID)
        assert hits, f"{who} should find their own paper"
        assert all(expect in h.chunk.text.lower() for h in hits)

    assert s.search("carol", "kidney results", HYBRID) == []
    assert s.search(None, "kidney results", HYBRID) == []


def test_dropping_a_session_removes_its_papers():
    s = store()
    s.add("alice", paper("kidney", "Kidney Segmentation At Scale"), "a.pdf")
    assert s.search("alice", "kidney", HYBRID)
    s.drop("alice")
    assert s.search("alice", "kidney", HYBRID) == []
    assert s.documents("alice") == []


def test_results_are_marked_as_uploaded():
    s = store()
    s.add("alice", paper("kidney", "Kidney Segmentation At Scale"), "a.pdf")
    hits = s.search("alice", "kidney", HYBRID)
    assert all("upload" in h.sources for h in hits)


def test_a_selection_of_only_corpus_papers_returns_nothing_from_uploads():
    s = store()
    doc = s.add("alice", paper("kidney", "Kidney Segmentation At Scale"), "a.pdf")
    assert s.search("alice", "kidney", HYBRID, doc_ids={"somecorpusdoc"}) == []
    assert s.search("alice", "kidney", HYBRID, doc_ids={doc.doc_id})


# ---- guards ----------------------------------------------------------------

def test_a_non_pdf_is_refused_by_content_not_extension():
    s = store()
    with pytest.raises(UploadError, match="not a PDF"):
        s.add("alice", b"PK\x03\x04 this is a docx", "paper.pdf")


def test_an_empty_file_is_refused():
    with pytest.raises(UploadError, match="empty"):
        store().add("alice", b"", "paper.pdf")


def test_an_oversize_file_is_refused_before_parsing():
    s = store()
    with pytest.raises(UploadError, match="limit"):
        s.add("alice", b"%PDF-" + b"\0" * (21 * 1024 * 1024), "big.pdf")


def test_a_scanned_pdf_says_so():
    """The parser distinguishes 'no text layer' from 'no sections'; the reader
    needs that distinction to know OCR is the fix."""
    s = store()
    with pytest.raises(UploadError, match="scanned"):
        s.add("alice", make_pdf(["x"], [11.0]), "scan.pdf")


def test_the_same_paper_twice_is_refused_not_indexed_twice():
    s = store()
    raw = paper("kidney", "Kidney Segmentation At Scale")
    s.add("alice", raw, "a.pdf")
    with pytest.raises(UploadError, match="already open"):
        s.add("alice", raw, "same-paper-different-name.pdf")


def test_a_paper_already_in_the_corpus_is_refused():
    raw = paper("kidney", "Kidney Segmentation At Scale")
    known = store()
    doc = known.add("probe", raw, "a.pdf")

    s = store()
    s.corpus_doc_ids = {doc.doc_id}
    with pytest.raises(UploadError, match="already in the indexed corpus"):
        s.add("alice", raw, "a.pdf")


def test_a_session_is_capped():
    s = store(max_papers=2)
    for i in range(2):
        s.add("alice", paper("kidney", f"Kidney Segmentation Volume {i}"), f"{i}.pdf")
    with pytest.raises(UploadError, match="limit is 2"):
        s.add("alice", paper("kidney", "Kidney Segmentation Volume 3"), "3.pdf")


# ---- eviction --------------------------------------------------------------

def test_idle_sessions_expire():
    s = store(ttl=0.0)
    s.add("alice", paper("kidney", "Kidney Segmentation At Scale"), "a.pdf")
    # The next upload runs eviction, and alice has been idle for longer than
    # a zero-second TTL.
    s.add("bob", paper("widget", "Widget Calibration In Practice"), "b.pdf")
    assert s.documents("alice") == []
    assert s.documents("bob")


def test_only_so_many_sessions_are_kept():
    s = store(max_sessions=2)
    for who in ("a", "b", "c"):
        s.add(who, paper("kidney", f"Kidney Segmentation By {who.upper()}"), f"{who}.pdf")
    live = [w for w in ("a", "b", "c") if s.documents(w)]
    assert len(live) == 2
    assert "c" in live, "the newest session must survive"
