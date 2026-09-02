"""The shared query layer, and the limitations path it feeds.

These are the regressions for a question that failed in the wild: "recent
research scope in LLM" searched all three indexes for the literal word *scope*,
because the three stop lists that existed at the time did not contain it and
did not contain each other either.
"""

import re

import pytest

from researchlens.engine import Engine
from researchlens.live import arxiv, openalex, pubmed, query
from researchlens.types import Chunk, Retrieved


def _r(text: str, heading: str = "Discussion", kind: str = "discussion") -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id="x:1", doc_id="x", ordinal=0, text=text,
            section_kind=kind, section_heading=heading,
            page_start=1, page_end=1, doc_title="A paper",
        ),
        score=0.0,
    )


# --- scaffolding ------------------------------------------------------------

def test_scope_is_not_a_topic():
    """The bug this module exists for. Asked the scope of recent LLM research,
    arXiv returned "SCOPE: A Generative Approach for LLM Prompt Compression"
    and PubMed returned the global burden of 292 causes of death."""
    assert "scope" not in query.terms("recent research scope in llm")


@pytest.mark.parametrize("word", [
    "recent", "current", "trends", "latest", "emerging",      # recency
    "scope", "overview", "landscape", "survey", "areas",      # shape
    "research", "papers", "literature", "studies", "field",   # the literature
    "tell", "me", "please", "show",                           # the system
])
def test_question_scaffolding_is_dropped(word):
    assert word not in query.terms(f"what are the {word} in diabetic retinopathy")


def test_the_subject_survives():
    got = query.terms("what are the recent research trends in diabetic retinopathy screening")
    assert "diabetic" in got and "retinopathy" in got and "screening" in got


def test_two_letter_acronyms_are_kept():
    """Requiring three characters dropped "AI" from "open problems in AI for
    radiology", leaving "radiology" alone — which, sorted by date, returned
    tantalum implants."""
    assert "ai" in query.terms("what are the major open problems in AI for radiology")


def test_terms_are_deduplicated_in_order():
    got = query.terms("llm and llm evaluation of llm systems")
    assert got.count("llm") == 1
    assert got.index("llm") < got.index("evaluation")


# --- acronyms ---------------------------------------------------------------

def test_an_acronym_gains_its_long_form():
    assert query.expand("llm") == ["llm", "large language model"]


def test_an_ordinary_word_is_left_alone():
    assert query.expand("retinopathy") == ["retinopathy"]


def test_expansion_never_replaces():
    """An index that only has the short form must still match."""
    assert query.expand("rag")[0] == "rag"


@pytest.mark.parametrize("ambiguous", ["dr", "ml", "ct", "us"])
def test_colliding_acronyms_are_not_expanded(ambiguous):
    """"dr" is diabetic retinopathy here and "doctor" everywhere else. Expanding
    it would cost more than it bought."""
    assert query.expand(ambiguous) == [ambiguous]


# --- each index in its own language -----------------------------------------

def test_arxiv_ors_an_expansion_rather_than_anding_it():
    """arXiv ANDs its terms, so `all:"llm" AND all:"large language model"`
    demands both — narrower than the acronym alone, which is the opposite of
    what expanding is for."""
    q = arxiv.build_query("recent research scope in llm")
    assert " OR " in q
    assert " AND " not in q


def test_pubmed_ors_an_expansion_inside_one_term():
    q = pubmed.build_query("recent research scope in llm")
    assert q.startswith("(") and " OR " in q


def test_openalex_appends_because_extra_terms_widen():
    """A ranked full-text match treats an extra term as extra evidence, so
    both forms are simply listed."""
    q = openalex.build_query("recent research scope in llm")
    assert "llm" in q and "large language model" in q
    assert " OR " not in q


def test_every_builder_drops_the_same_scaffolding():
    """The failure was three lists disagreeing. They cannot now."""
    q = "what are the recent research trends in llm"
    for built in (arxiv.build_query(q), pubmed.build_query(q), openalex.build_query(q)):
        assert "trend" not in built.lower()
        assert "research" not in built.lower()


def test_a_question_with_no_subject_searches_for_nothing():
    """Every source used to invent one — arXiv fell back to "machine learning"
    and PubMed to "biomedical research" — which searched for something the
    reader had not asked about and returned abstracts that looked like
    evidence. An empty query is the honest answer, and each source declines."""
    for built in (arxiv.build_query("what are the recent trends"),
                  pubmed.build_query("what are the recent trends"),
                  openalex.build_query("tell me about recent research")):
        assert built == ""


# --- limitations ------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "what limitations do the authors state for RAG?",
    "what are the drawbacks of this approach",
    "what are the weaknesses of deep learning for screening",
    "what caveats do the authors mention",
    "where does this method fail",
    "what future work do they suggest",
    "what cannot this model do",
])
def test_limitation_questions_are_recognised(q):
    assert query.asks_for_limitations(q)


@pytest.mark.parametrize("q", [
    "how does retrieval-augmented generation reduce hallucination?",
    "what datasets were used for training",
    "how do vision-language models align image and text representations?",
])
def test_ordinary_questions_are_not(q):
    assert not query.asks_for_limitations(q)


def test_a_heading_naming_limitations_counts():
    assert Engine._states_a_limitation(_r("Anything.", "Conclusion & Limitation"))


def test_a_concession_in_prose_counts():
    """Most papers have no limitations section and concede in prose instead."""
    assert Engine._states_a_limitation(
        _r("We do not evaluate on external cohorts, which remains an open question.")
    )


def test_a_caption_does_not_count():
    """A caption saying a method "fails on" one of six examples is describing a
    figure, not stating the bounds of the work. 165 of this corpus's 678 cue
    matches are captions and table cells."""
    assert not Engine._states_a_limitation(
        _r("The method fails on example 3.", "Figure 3: Six examples", "figure")
    )
    assert not Engine._states_a_limitation(
        _r("Model fails to converge.", "Table 2", "table")
    )


def test_a_page_limit_is_not_a_research_limitation():
    """The lexical false friend that led an answer on the live Space.

    An AutoGen caption reading "Due to the page limit, details of the
    evaluation are in Appendix D" scored +5.34 against "what limitations do
    the authors state" — more than double the next passage — because a
    cross-encoder sees "limit". A page limit is a fact about a conference
    template, not about the work. Captions leave the pool entirely on this
    question, which is what keeps it out.
    """
    caption = _r(
        "Due to the page limit, details of the evaluation, including case "
        "studies in three scenarios, are in Appendix D.",
        "Figure 3: Six examples of diverse applications",
        "figure",
    )
    assert not Engine._states_a_limitation(caption)
    assert caption.chunk.section_kind in Engine._NOT_A_CONCESSION


def test_an_ordinary_passage_does_not_count():
    assert not Engine._states_a_limitation(
        _r("We train a ConvNeXt V2 backbone on six public datasets.")
    )


def test_the_prompt_asks_for_the_authors_own_words():
    from researchlens.generate.prompt import build_prompt

    _sys, user = build_prompt(
        "what limitations do the authors state for RAG?",
        [_r("We do not evaluate on external cohorts.")],
    )
    assert "authors' own terms" in user
    # The rule that matters: no inventing weaknesses that nobody wrote down.
    assert re.search(r"do not add weaknesses of your own", user, re.I)


def test_an_ordinary_question_gets_no_limitations_clause():
    """A rule in the prompt is paid for by every question that does not need
    it — nine lines added to SYSTEM once stopped a 3B model citing at all."""
    from researchlens.generate.prompt import build_prompt

    _sys, user = build_prompt(
        "how does RAG reduce hallucination?",
        [_r("RAG retrieves passages before generating.")],
    )
    assert "authors' own terms" not in user


# --- comparison -------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "compare my paper with the recent online literature",
    "how does my paper differ from what is on arxiv",
    "what is the difference between this and the indexed papers",
    "how does this contrast with the state of the art",
])
def test_comparison_questions_are_recognised(q):
    assert query.asks_to_compare(q)


def test_a_comparison_naming_the_outside_is_distinguished():
    """"Compare this with the recent literature" is a different request from
    "compare this with the indexed papers", and answering the first from a
    fixed corpus is a confident comparison against the wrong evidence."""
    assert query.compares_with_online("compare my paper with the recent online literature")
    assert not query.compares_with_online("compare this with the indexed papers")


def test_comparison_words_are_not_searched_for():
    """"compare my paper with the recent online literature on agentic RAG"
    searched arXiv for `compare AND online AND agentic AND rag`, which returned
    a video-generation paper. The subject was two words of that sentence."""
    assert query.terms(
        "compare my paper with the recent online literature on agentic RAG"
    ) == ["agentic", "rag"]


def test_the_prompt_names_each_side_of_a_comparison():
    from researchlens.generate.prompt import build_prompt

    ev = [
        _r("Mine.", "Methods"),
        _r("An abstract.", "arXiv 1 · abstract"),
        _r("A corpus passage.", "Results"),
    ]
    # Tag the first as the reader's own, the way uploads.py does.
    ev[0] = Retrieved(chunk=ev[0].chunk, score=0.0, sources=frozenset({"upload"}))
    ev[1] = Retrieved(
        chunk=Chunk(
            chunk_id="arxiv:1", doc_id="arxiv:1", ordinal=0, text="An abstract.",
            section_kind="abstract", section_heading="arXiv 1", page_start=0,
            page_end=0, doc_title="A preprint",
        ),
        score=0.0,
    )
    _sys, user = build_prompt("compare my paper with the online literature", ev)
    assert "asks for a comparison" in user
    assert "reader's own uploaded paper" in user
    assert "1 abstract fetched" in user          # singular, not "1 abstracts"
    assert "indexed corpus" in user


def test_an_ordinary_question_gets_no_comparison_clause():
    from researchlens.generate.prompt import build_prompt

    _sys, user = build_prompt("what is agentic RAG?", [_r("A passage.")])
    assert "asks for a comparison" not in user
