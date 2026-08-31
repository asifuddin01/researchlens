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
