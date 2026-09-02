"""The author's added library — fetching, indexing and merging.

No network and no PDFs. `parse_bytes` is stubbed, because what is under test
here is everything around the parser: which papers survive the manifest, what
happens when one of them will not parse, and whether an added paper competes
with the corpus rather than displacing it. The parser has its own tests, and
the live endpoint is checked by scripts/verify.py.
"""

import asyncio

import httpx
import pytest

from researchlens.engine import Engine
from researchlens.library import LibraryIndex
from researchlens.live import library as source
from researchlens.types import Chunk, Document, Retrieved, Section


MANIFEST = {
    "author": "Md. Asif Uddin",
    "site": "https://asifuddin.com",
    "schema": 1,
    "documents": [
        {
            "id": "library:radii",
            "title": "Rank Radii Transfer as Quantiles",
            "authors": "Md. Asif Uddin",
            "year": "2026",
            "note": "",
            "url": "https://asifuddin.com/library/radii.pdf",
            "page": "https://asifuddin.com/papers#library",
        },
        {
            "id": "library:scoping",
            "title": "A Scoping Review of HCGT-PG",
            "authors": "Md. Asif Uddin",
            "year": "2026",
            "note": "",
            "url": "https://asifuddin.com/library/scoping.pdf",
            "page": "https://asifuddin.com/papers#library",
        },
    ],
}


def _doc(doc_id: str, title: str) -> Document:
    return Document(
        doc_id=doc_id, title=title, authors=["Md. Asif Uddin"],
        sections=[Section(kind="methods", heading="Methods",
                          text="Radii are transferred as quantiles. " * 30,
                          page_start=1, page_end=2)],
        n_pages=2, source_path=f"library:{doc_id}",
    )


@pytest.fixture(autouse=True)
def _clean():
    source.reset_cache()
    yield
    source.reset_cache()


def _client(handler):
    class _Fake:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            return handler(url, headers or {})

    return _Fake


def _serve(manifest=MANIFEST, pdf=b"%PDF-1.4 fake", etag="e1"):
    """A manifest at the manifest URL, and bytes at every PDF URL."""
    def handler(url, headers):
        req = httpx.Request("GET", url)
        if url.endswith(".pdf"):
            return httpx.Response(200, content=pdf, request=req)
        return httpx.Response(200, json=manifest, headers={"ETag": etag}, request=req)
    return handler


def _parses_as(mapping):
    """Stub the parser: PDF bytes in, a named Document out."""
    calls = {"n": 0}

    def parse_bytes(raw, name, source=None):
        calls["n"] += 1
        if name not in mapping:
            raise ValueError(f"no text layer in {name}")
        return mapping[name]

    return parse_bytes, calls


# --- fetching ---------------------------------------------------------------

def test_a_listed_paper_becomes_passages(monkeypatch):
    parse, _ = _parses_as({
        "Rank Radii Transfer as Quantiles": _doc("aaa", "Rank Radii Transfer as Quantiles"),
        "A Scoping Review of HCGT-PG": _doc("bbb", "A Scoping Review of HCGT-PG"),
    })
    monkeypatch.setattr(source, "parse_bytes", parse)
    monkeypatch.setattr(httpx, "AsyncClient", _client(_serve()))

    docs, chunks = asyncio.run(source.fetch())
    assert {d.doc_id for d in docs} == {"aaa", "bbb"}
    assert chunks and source.last_error is None


def test_the_authors_own_title_wins_over_the_parsers_guess(monkeypatch):
    """The parser infers a title from typography and guesses badly on anything
    that is not a typeset journal page — given a plain export it took the whole
    opening paragraph. Here nobody has to guess: it was typed into the CMS."""
    parse, _ = _parses_as({
        "Rank Radii Transfer as Quantiles": _doc(
            "aaa", "Rank Radii Transfer as Quantiles Abstract We study the transfer of"
        ),
    })
    monkeypatch.setattr(source, "parse_bytes", parse)
    monkeypatch.setattr(httpx, "AsyncClient", _client(_serve()))

    docs, chunks = asyncio.run(source.fetch())
    assert docs[0].title == "Rank Radii Transfer as Quantiles"
    assert all(c.doc_title == "Rank Radii Transfer as Quantiles" for c in chunks)


def test_the_authors_come_from_the_cms_too(monkeypatch):
    """Nothing in a PDF marks a byline as a byline, so the parser usually
    finds none — and the source browser then lists a paper by nobody."""
    doc = _doc("aaa", "R")
    from dataclasses import replace
    parse, _ = _parses_as({"Rank Radii Transfer as Quantiles": replace(doc, authors=[])})
    monkeypatch.setattr(source, "parse_bytes", parse)
    monkeypatch.setattr(httpx, "AsyncClient", _client(_serve()))

    docs, _ = asyncio.run(source.fetch())
    assert docs[0].authors == ["Md. Asif Uddin"]


def test_a_citation_links_to_the_pdf(monkeypatch):
    """The corpus cannot link its papers — they are licensed journal PDFs.
    These are on the author's own site, so a citation can be a link."""
    parse, _ = _parses_as({"Rank Radii Transfer as Quantiles": _doc("aaa", "R")})
    monkeypatch.setattr(source, "parse_bytes", parse)
    monkeypatch.setattr(httpx, "AsyncClient", _client(_serve()))

    _, chunks = asyncio.run(source.fetch())
    assert chunks
    assert all(c.url == "https://asifuddin.com/library/radii.pdf" for c in chunks)


def test_a_paper_that_will_not_parse_is_named_not_swallowed(monkeypatch):
    """A scan indexes as nothing. Silence would make it indistinguishable from
    a paper the corpus simply has nothing to say about."""
    parse, _ = _parses_as({"Rank Radii Transfer as Quantiles": _doc("aaa", "R")})
    monkeypatch.setattr(source, "parse_bytes", parse)
    monkeypatch.setattr(httpx, "AsyncClient", _client(_serve()))

    docs, _ = asyncio.run(source.fetch())
    assert [d.doc_id for d in docs] == ["aaa"]
    assert any("Scoping" in s for s in source.skipped), source.skipped


def test_one_broken_paper_does_not_cost_the_others(monkeypatch):
    parse, _ = _parses_as({"A Scoping Review of HCGT-PG": _doc("bbb", "S")})
    monkeypatch.setattr(source, "parse_bytes", parse)
    monkeypatch.setattr(httpx, "AsyncClient", _client(_serve()))

    docs, chunks = asyncio.run(source.fetch())
    assert [d.doc_id for d in docs] == ["bbb"] and chunks


def test_an_oversized_pdf_is_refused_by_size(monkeypatch):
    parse, calls = _parses_as({"Rank Radii Transfer as Quantiles": _doc("aaa", "R")})
    monkeypatch.setattr(source, "parse_bytes", parse)
    monkeypatch.setattr(source, "MAX_BYTES", 8)
    monkeypatch.setattr(httpx, "AsyncClient", _client(_serve(pdf=b"%PDF larger than eight")))

    docs, _ = asyncio.run(source.fetch())
    assert docs == []
    assert calls["n"] == 0, "an oversized file should not reach the parser"
    assert any("larger than" in s for s in source.skipped)


def test_the_manifest_is_capped(monkeypatch):
    many = {**MANIFEST, "documents": MANIFEST["documents"] * 20}
    parse, calls = _parses_as({})
    monkeypatch.setattr(source, "parse_bytes", parse)
    monkeypatch.setattr(source, "MAX_PAPERS", 3)
    monkeypatch.setattr(httpx, "AsyncClient", _client(_serve(many)))

    asyncio.run(source.fetch())
    assert calls["n"] <= 3


def test_a_second_question_inside_the_ttl_costs_nothing(monkeypatch):
    seen = []
    parse, _ = _parses_as({"Rank Radii Transfer as Quantiles": _doc("aaa", "R")})
    monkeypatch.setattr(source, "parse_bytes", parse)

    inner = _serve()

    def handler(url, headers):
        seen.append(url)
        return inner(url, headers)

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    asyncio.run(source.fetch())
    n = len(seen)
    asyncio.run(source.fetch())
    assert len(seen) == n


def test_an_unchanged_manifest_does_not_redownload(monkeypatch):
    """The point of the short TTL: checking is one conditional request, and
    only a manifest that actually changed pays for downloads and parsing."""
    state = {"n": 0}
    parse, calls = _parses_as({"Rank Radii Transfer as Quantiles": _doc("aaa", "R")})
    monkeypatch.setattr(source, "parse_bytes", parse)
    inner = _serve()

    def handler(url, headers):
        if url.endswith(".pdf"):
            return inner(url, headers)
        state["n"] += 1
        if state["n"] == 1:
            return inner(url, headers)
        assert headers.get("If-None-Match") == "e1", "should revalidate"
        return httpx.Response(304, headers={"ETag": "e1"},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    asyncio.run(source.fetch())
    parsed = calls["n"]
    kept = asyncio.run(source.fetch(force=True))
    assert calls["n"] == parsed, "a 304 should not re-parse anything"
    assert kept[1], "and should keep the passages"


def test_a_stale_library_beats_no_library(monkeypatch):
    state = {"n": 0}
    parse, _ = _parses_as({"Rank Radii Transfer as Quantiles": _doc("aaa", "R")})
    monkeypatch.setattr(source, "parse_bytes", parse)
    inner = _serve()

    def handler(url, headers):
        if not url.endswith(".pdf"):
            state["n"] += 1
            if state["n"] > 1:
                raise httpx.ConnectError("gone")
        return inner(url, headers)

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    n = len(asyncio.run(source.fetch())[1])
    kept = asyncio.run(source.fetch(force=True))
    assert len(kept[1]) == n
    assert source.last_error is not None


def test_a_failed_fetch_is_reported(monkeypatch):
    def handler(url, headers):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    assert asyncio.run(source.fetch()) == ([], [])
    assert source.last_error and "ConnectError" in source.last_error


# --- the index --------------------------------------------------------------

class _Reranker:
    """Scores by keyword overlap. Enough to order a pool deterministically."""

    def rerank(self, query, chunks, k):
        want = set(query.lower().split())
        scored = [
            (c.chunk_id, float(len(want & set(c.text.lower().split()))))
            for c in chunks
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]


def _index(monkeypatch, mapping, corpus_doc_ids=None):
    parse, _ = _parses_as(mapping)
    monkeypatch.setattr(source, "parse_bytes", parse)
    monkeypatch.setattr(httpx, "AsyncClient", _client(_serve()))
    idx = LibraryIndex(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker=_Reranker(),
        corpus_doc_ids=corpus_doc_ids or set(),
    )
    return idx


def test_a_paper_already_in_the_corpus_is_not_indexed_twice(monkeypatch):
    """Two copies of one paper would let a single source fill two evidence
    slots and read to the model as corroboration."""
    idx = _index(
        monkeypatch,
        {"Rank Radii Transfer as Quantiles": _doc("aaa", "R"),
         "A Scoping Review of HCGT-PG": _doc("bbb", "S")},
        corpus_doc_ids={"aaa"},
    )
    asyncio.run(idx.refresh())
    assert idx.doc_ids == {"bbb"}
    assert any("already in the indexed corpus" in s for s in idx.skipped)


def test_the_same_pdf_listed_twice_does_not_break_the_index(monkeypatch):
    """`RetrievalPipeline.index` refuses duplicate chunk ids outright, so an
    unfiltered manifest would take down the library rather than one entry."""
    same = _doc("aaa", "R")
    idx = _index(
        monkeypatch,
        {"Rank Radii Transfer as Quantiles": same,
         "A Scoping Review of HCGT-PG": same},
    )
    asyncio.run(idx.refresh())
    assert len(idx.documents) == 1
    ids = [c.chunk_id for c in idx.chunks]
    assert len(ids) == len(set(ids))
    assert any("more than once" in s for s in idx.skipped)


def test_an_unchanged_library_is_not_reindexed(monkeypatch):
    idx = _index(monkeypatch, {"Rank Radii Transfer as Quantiles": _doc("aaa", "R")})
    asyncio.run(idx.refresh())
    first = idx.pipeline
    assert first is not None
    asyncio.run(idx.refresh(force=True))
    assert idx.pipeline is first, "same passages should not cost an embedding pass"


def test_a_failed_fetch_keeps_the_previous_index(monkeypatch):
    idx = _index(monkeypatch, {"Rank Radii Transfer as Quantiles": _doc("aaa", "R")})
    asyncio.run(idx.refresh())
    assert idx.documents

    def dead(url, headers):
        raise httpx.ConnectError("gone")

    monkeypatch.setattr(httpx, "AsyncClient", _client(dead))
    source.reset_cache()
    asyncio.run(idx.refresh(force=True))
    assert idx.documents, "a stale paper answers better than no paper"
    assert idx.last_error


def test_an_empty_library_costs_nothing_to_search(monkeypatch):
    idx = LibraryIndex(embedding_model="x", reranker=_Reranker())
    from researchlens.engine import SERVING
    assert idx.search("anything", SERVING) == []


# --- merging ----------------------------------------------------------------

def _r(chunk_id: str, doc_id: str, score: float) -> Retrieved:
    return Retrieved(
        chunk=Chunk(chunk_id=chunk_id, doc_id=doc_id, ordinal=0, text="t",
                    section_kind="methods", section_heading="Methods",
                    page_start=1, page_end=1, doc_title="T"),
        score=score, rerank_score=score,
    )


def test_an_added_paper_competes_rather_than_displacing():
    """No reserved slots, unlike every other merge: an added paper is a paper,
    and reserving room would reward the accident of when it was uploaded."""
    e = Engine.__new__(Engine)
    added = [_r("l1", "lib", 5.0), _r("l2", "lib", 0.1)]
    corpus = [_r("c1", "a", 4.0), _r("c2", "b", 3.0), _r("c3", "c", 2.0)]
    out = e._merge_library(added, corpus, 4)
    assert [r.chunk.chunk_id for r in out] == ["l1", "c1", "c2", "c3"]


def test_an_added_paper_cannot_take_the_whole_answer():
    """Six of eight passages from one paper is a paraphrase wearing the
    question's clothes — the failure `_diversify` exists for."""
    e = Engine.__new__(Engine)
    added = [_r(f"l{i}", "lib", 9.0 - i) for i in range(6)]
    corpus = [_r("c1", "a", 1.0), _r("c2", "b", 0.5)]
    out = e._merge_library(added, corpus, 8)
    assert sum(1 for r in out if r.chunk.doc_id == "lib") == 2


def test_an_irrelevant_added_paper_cannot_backfill_the_corpus():
    """The bug this merge was rewritten for. Capping the union re-ran the
    corpus's own diversity rule, discarded corpus passages, and then filled the
    slots it had just freed with the library — seating a paper on renal radii
    at -11.25 in a question about retrieval-augmented generation."""
    e = Engine.__new__(Engine)
    # Two papers, three passages each: a union-wide cap of two would drop the
    # third of each and leave two slots for anything at all.
    corpus = [_r(f"c{i}", "a" if i < 3 else "b", 5.0 - i) for i in range(6)]
    added = [_r("l1", "lib", -11.25), _r("l2", "lib", -11.38)]
    out = e._merge_library(added, corpus, 8)
    assert [r.chunk.chunk_id for r in out][:6] == [f"c{i}" for i in range(6)]
    assert not any(r.chunk.doc_id == "lib" for r in out[:6])


def test_a_weak_added_paper_does_not_reach_the_answer():
    e = Engine.__new__(Engine)
    added = [_r("l1", "lib", -3.0)]
    corpus = [_r(f"c{i}", chr(97 + i), 5.0 - i) for i in range(4)]
    out = e._merge_library(added, corpus, 4)
    assert "l1" not in [r.chunk.chunk_id for r in out]
