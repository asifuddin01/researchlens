"""Live-search tests.

Network calls are not made here — these cover the query building and the merge,
which is where the logic is. The sources themselves are exercised by
scripts/verify.py against the real APIs.
"""

import asyncio

from researchlens.live import arxiv, pubmed, search
from researchlens.live.arxiv import LivePaper, is_live, to_chunks


def _paper(title: str, source: str = "arxiv", pid: str = "1") -> LivePaper:
    return LivePaper(
        paper_id=pid, title=title, authors=["A"], abstract="An abstract.",
        published="2026-08-01", url="https://example.org", source=source,
    )


# --- query building ---------------------------------------------------------

def test_arxiv_query_drops_question_scaffolding():
    q = arxiv.build_query("What are the major current research trends in large language models?")
    assert "large" in q and "language" in q
    assert "trends" not in q and "what" not in q


def test_pubmed_keeps_short_acronyms():
    """Requiring three characters dropped "AI" from "AI for radiology", leaving
    the single term "radiology" — which returned tantalum implants."""
    q = pubmed.build_query("What are the major open problems in AI for radiology?")
    assert "ai" in q.split(" AND ")
    assert "radiology" in q


def test_a_query_with_only_stopwords_still_produces_something():
    assert arxiv.build_query("what are the current trends?")
    assert pubmed.build_query("what are the current trends?")


# --- adapting to chunks -----------------------------------------------------

def test_live_chunks_are_identifiable_and_pageless():
    c = to_chunks([_paper("A Title")])[0]
    assert is_live(c.chunk_id)
    assert c.pages == "abstract"
    assert "2026-08-01" in c.section_heading


# --- merging ----------------------------------------------------------------

def test_sources_are_interleaved_not_concatenated():
    """Whichever source is read first gets cited more, so ordering is an
    editorial decision and must not fall out of dict order."""
    async def a(*_a, **_k):
        return [_paper("Alpha", "arxiv", "a1"), _paper("Beta", "arxiv", "a2")]

    async def b(*_a, **_k):
        return [_paper("Gamma", "pubmed", "p1"), _paper("Delta", "pubmed", "p2")]

    original = dict(search.SOURCES)
    search.SOURCES.clear()
    search.SOURCES.update({"arxiv": a, "pubmed": b})
    try:
        got = asyncio.run(search.search("q"))
        assert [p.source for p in got] == ["arxiv", "pubmed", "arxiv", "pubmed"]
    finally:
        search.SOURCES.clear()
        search.SOURCES.update(original)


def test_the_same_work_from_two_sources_appears_once():
    """A paper is routinely both an arXiv preprint and a journal article."""
    async def a(*_a, **_k):
        return [_paper("Attention Is All You Need", "arxiv", "a1")]

    async def b(*_a, **_k):
        return [_paper("Attention is all you need!", "pubmed", "p1")]

    original = dict(search.SOURCES)
    search.SOURCES.clear()
    search.SOURCES.update({"arxiv": a, "pubmed": b})
    try:
        assert len(asyncio.run(search.search("q"))) == 1
    finally:
        search.SOURCES.clear()
        search.SOURCES.update(original)


def test_one_source_failing_does_not_lose_the_other():
    """Live search is an enhancement: a PubMed-only answer beats an error."""
    async def ok(*_a, **_k):
        return [_paper("Survivor", "pubmed", "p1")]

    async def boom(*_a, **_k):
        raise RuntimeError("network down")

    original = dict(search.SOURCES)
    search.SOURCES.clear()
    search.SOURCES.update({"arxiv": boom, "pubmed": ok})
    try:
        got = asyncio.run(search.search("q"))
        assert [p.title for p in got] == ["Survivor"]
    finally:
        search.SOURCES.clear()
        search.SOURCES.update(original)


def test_no_known_source_returns_nothing_rather_than_raising():
    assert asyncio.run(search.search("q", sources=["scopus"])) == []


# ---- OpenAlex --------------------------------------------------------------

def test_an_inverted_abstract_is_rebuilt_in_order():
    from researchlens.live.openalex import reconstruct_abstract

    inverted = {"Federated": [0], "learning": [1, 5], "enables": [2],
                "private": [3], "collaborative": [4], ".": [6]}
    assert reconstruct_abstract(inverted) == (
        "Federated learning enables private collaborative learning ."
    )


def test_a_missing_inverted_abstract_is_empty_not_an_error():
    from researchlens.live.openalex import reconstruct_abstract

    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""


def test_openalex_keeps_more_terms_than_pubmed():
    """OpenAlex ranks a full-text match, so extra terms reorder rather than
    narrow to nothing — the opposite of PubMed's boolean AND."""
    from researchlens.live import openalex, pubmed

    q = "What are the current trends in federated learning for medical imaging?"
    assert openalex.build_query(q) == "federated learning medical imaging"
    assert " AND " in pubmed.build_query(q)


def test_every_live_source_has_a_display_name():
    """A citation heading is built from this map; a source missing from it
    would be cited by its internal key."""
    from researchlens.live.arxiv import LIVE_PREFIXES, SOURCE_NAMES
    from researchlens.live.search import SOURCES

    assert set(SOURCES) <= set(LIVE_PREFIXES)
    assert set(SOURCES) <= set(SOURCE_NAMES)


def test_a_live_chunk_names_its_index_and_cannot_pass_as_a_page():
    from researchlens.live.arxiv import LivePaper, is_live, to_chunks

    chunks = to_chunks([
        LivePaper(paper_id="10.1109/TMI.2024.1", title="A Journal Paper",
                  authors=["R Lee"], abstract="[IEEE TMI] We measured things.",
                  published="2024-10-18", url="https://doi.org/x",
                  source="openalex"),
    ])
    c = chunks[0]
    assert is_live(c.chunk_id)
    assert c.pages == "abstract"
    assert "OpenAlex" in c.section_heading


# ---- the survey instruction must describe the evidence it is sent with -----

def _retrieved(chunk_id: str, title: str, text: str = "Some findings."):
    from researchlens.types import Chunk, Retrieved

    doc = chunk_id.split(":")[0] if ":" in chunk_id else chunk_id.split("-")[0]
    return Retrieved(
        chunk=Chunk(
            chunk_id=chunk_id, doc_id=doc, ordinal=0, text=text,
            section_kind="abstract", section_heading="h",
            page_start=0, page_end=0, doc_title=title,
        ),
        score=1.0,
    )


def test_a_survey_question_with_live_evidence_is_told_the_evidence_is_recent():
    """The instruction went stale the day live search landed: it told the model
    its evidence was a fixed corpus even when abstracts had been fetched
    minutes earlier, and the model duly refused with three of them in hand."""
    from researchlens.generate.prompt import build_prompt

    _system, user = build_prompt(
        "What are the current trends in long-context language models?",
        [_retrieved("arxiv:2608.1", "A Recent Paper"),
         _retrieved("abc123:4", "An Indexed Paper")],
    )
    assert "fetched" in user
    assert "fixed set of indexed papers rather than a survey" not in user


def test_a_survey_question_without_live_evidence_still_says_the_corpus_is_fixed():
    from researchlens.generate.prompt import build_prompt

    _system, user = build_prompt(
        "What are the current trends in long-context language models?",
        [_retrieved("abc123:4", "An Indexed Paper")],
    )
    assert "fixed set of indexed papers" in user
    assert "fetched" not in user


def test_an_ordinary_question_gets_no_survey_instruction():
    from researchlens.generate.prompt import build_prompt

    _system, user = build_prompt(
        "What datasets does scGPT use?", [_retrieved("abc123:4", "An Indexed Paper")]
    )
    assert "fixed set of indexed papers" not in user
    assert "fetched" not in user
