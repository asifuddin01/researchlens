"""Citation resolution tests.

The behaviour under test is the project's central claim: what reaches the
reader is checked, not reported.
"""

from researchlens.generate.citations import is_grounded, resolve
from researchlens.types import Chunk, Retrieved


def _ev(n: int) -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=f"doc:{n}", doc_id="doc", ordinal=n, text=f"Passage {n} body text.",
            section_kind="results", section_heading="Results",
            page_start=n, page_end=n, doc_title=f"Paper {n}",
        ),
        score=1.0,
    )


EV = [_ev(1), _ev(2), _ev(3)]


def test_valid_markers_resolve_to_citations():
    text, cites = resolve("The model improved [1] and generalised [2].", EV)
    assert [c.marker for c in cites] == [1, 2]
    assert cites[0].doc_title == "Paper 1"
    assert cites[0].pages == "1"
    assert "[1]" in text and "[2]" in text


def test_an_invented_marker_is_removed():
    """A [7] pointing outside the evidence must never reach the reader."""
    text, cites = resolve("Supported claim [1]. Invented claim [7].", EV)
    assert "[7]" not in text
    assert len(cites) == 1


def test_survivors_are_renumbered_densely():
    """After removing a fabrication the reader should see [1][2], not [1][3]."""
    text, cites = resolve("First [2]. Second [9]. Third [3].", EV)
    assert "[1]" in text and "[2]" in text
    assert "[9]" not in text
    assert [c.marker for c in cites] == [1, 2]
    # Renumbering follows first appearance, not retrieval rank.
    assert cites[0].doc_title == "Paper 2"


def test_removing_a_marker_does_not_leave_stray_punctuation():
    text, _ = resolve("A claim [7] .", EV)
    assert " ." not in text


def test_a_repeated_marker_is_cited_once():
    _text, cites = resolve("Claim [1]. Related claim [1].", EV)
    assert len(cites) == 1


def test_an_answer_with_no_citation_is_not_grounded():
    text, cites = resolve("Transformers were introduced in 2017.", EV)
    assert cites == []
    assert not is_grounded(text, cites)


def test_an_explicit_refusal_is_grounded():
    """Refusing is the correct output and has nothing to cite."""
    text = "I could not find sufficient evidence in the indexed papers."
    assert is_grounded(text, [])


def test_a_quote_is_carried_for_every_citation():
    _text, cites = resolve("Claim [3].", EV)
    assert cites[0].quote.startswith("Passage 3")


# --- survey scoping ---------------------------------------------------------

from researchlens.generate.prompt import asks_for_a_survey, build_prompt  # noqa: E402


def test_a_trend_question_is_recognised_as_a_survey():
    """Observed failure: the system answered "current research trends in large
    language models" from Attention Is All You Need (2017) and BERT (2018),
    with real citations and no sign that its evidence was eight years old."""
    for q in (
        "What are the major current research trends in large language models?",
        "What are the latest approaches to retrieval?",
        "Which method is state of the art?",
        "What research directions are emerging?",
    ):
        assert asks_for_a_survey(q), q


def test_an_ordinary_question_is_not_a_survey():
    for q in (
        "What dataset did scGPT use?",
        "Do deep-learning models outperform linear baselines?",
        "What are the limitations the authors acknowledge?",
    ):
        assert not asks_for_a_survey(q), q


def test_a_survey_question_carries_a_scope_instruction():
    _sys, user = build_prompt("What are the current trends in RAG?", EV)
    assert "fixed set of indexed papers" in user


def test_an_ordinary_question_carries_no_scope_instruction():
    _sys, user = build_prompt("What dataset was used?", EV)
    assert "fixed set of indexed papers" not in user


def test_no_evidence_never_reaches_the_model():
    """Rule 5: a model asked a question with no context answers from training,
    fluently, and a reader cannot tell that from a grounded answer."""
    import pytest
    with pytest.raises(ValueError, match="no evidence"):
        build_prompt("anything", [])


# --- live evidence ----------------------------------------------------------

from researchlens.generate.prompt import build_context  # noqa: E402
from researchlens.live.arxiv import LivePaper, is_live, to_chunks  # noqa: E402


def _live_ev():
    papers = [LivePaper(
        paper_id="2608.28444v1", title="Sliding-window beats linear attention",
        authors=["A. Author"], abstract="We show that sliding-window attention...",
        published="2026-08-28", url="https://arxiv.org/abs/2608.28444v1",
    )]
    return [Retrieved(chunk=c, score=0.0) for c in to_chunks(papers)]


def test_a_live_result_is_identifiable_from_its_id():
    ev = _live_ev()
    assert is_live(ev[0].chunk.chunk_id)
    assert not is_live("93b9a09b1d7d2c9e:5")


def test_a_live_result_has_no_page_number():
    """Printing "p0" would invite a reader to look for a page that does not
    exist; an abstract has no page."""
    assert _live_ev()[0].chunk.pages == "abstract"


def test_the_prompt_marks_live_evidence_as_abstract_only():
    """Without this the model writes "the paper reports X on dataset Y" from an
    abstract that never said so."""
    ctx = build_context(_live_ev())
    assert "ABSTRACT ONLY" in ctx
    assert "2026-08-28" in ctx


def test_corpus_passages_are_not_marked_abstract_only():
    assert "ABSTRACT ONLY" not in build_context(EV)


def test_a_live_citation_names_its_source_and_date():
    _text, cites = resolve("Recent work shows this [1].", _live_ev())
    assert "arXiv 2608.28444v1" in cites[0].section_heading
    assert cites[0].pages == "abstract"


# ---- several sources in one bracket ----------------------------------------

def _ev(n: int):
    from researchlens.types import Chunk, Retrieved

    return [
        Retrieved(
            chunk=Chunk(
                chunk_id=f"d{i}:0", doc_id=f"d{i}", ordinal=0, text=f"Passage {i}.",
                section_kind="methods", section_heading="h",
                page_start=1, page_end=1, doc_title=f"Paper {i}",
            ),
            score=1.0,
        )
        for i in range(1, n + 1)
    ]


def test_a_grouped_marker_resolves_every_source_in_it():
    """A stronger model writes "[1, 2, 3]" naturally. A pattern matching only
    "[n]" left the whole span as literal text pointing at nothing while the
    other sources vanished from the evidence panel — a marker resolving to no
    passage, which is the failure this module exists to prevent."""
    text, cites = resolve("Baselines win [1, 2, 3, 4, 5, 6, 7].", _ev(7))
    assert len(cites) == 7
    assert text == "Baselines win [1][2][3][4][5][6][7]."


def test_a_semicolon_group_is_read_the_same_way():
    text, cites = resolve("Mixed [2; 4] and single [1].", _ev(5))
    assert [c.marker for c in cites] == [1, 2, 3]
    assert text == "Mixed [1][2] and single [3]."


def test_a_partly_invented_group_keeps_only_what_exists():
    """A group can be part real and part fabricated; each number is judged on
    its own rather than the bracket surviving or dying whole."""
    text, cites = resolve("Partly invented [2, 99].", _ev(3))
    assert len(cites) == 1
    assert text == "Partly invented [1]."


def test_a_wholly_invented_group_leaves_no_bracket():
    text, cites = resolve("All invented [88, 99].", _ev(3))
    assert cites == []
    assert text == "All invented."


def test_spacing_inside_a_bracket_does_not_matter():
    _text, cites = resolve("Spaced [ 1 , 2 ] here.", _ev(2))
    assert len(cites) == 2
