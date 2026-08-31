"""De-hyphenation tests.

Every case is a real string observed in the corpus. The rule is decided against
the document's own vocabulary, so each test supplies the vocabulary the paper
would have provided.
"""

from researchlens.ingest.parse import _dehyphenate, _document_vocabulary, _Line


def test_a_broken_word_is_rejoined():
    vocab = {"expression", "gene", "changes"}
    assert _dehyphenate("gene expres- sion changes", vocab) == "gene expression changes"


def test_a_real_compound_keeps_its_hyphen():
    """Both halves are words the document uses, so this is a compound broken at
    a line end, not a broken word."""
    vocab = {"single", "cell", "data"}
    assert _dehyphenate("single- cell data", vocab) == "single-cell data"


def test_the_hyphenated_form_wins_when_the_document_uses_it():
    vocab = {"single-cell", "single", "cell"}
    assert _dehyphenate("single- cell", vocab) == "single-cell"


def test_a_superscript_reference_marker_is_not_a_word_break():
    """'Two recent mod- 7 8 els' — the 7 is a citation marker, not a
    continuation."""
    vocab = {"models", "recent"}
    assert _dehyphenate("recent mod- 7", vocab) == "recent mod- 7"


def test_a_capitalised_continuation_is_not_a_word_break():
    vocab = {"gene", "gears"}
    assert _dehyphenate("Gene- GEARS", vocab) == "Gene- GEARS"


def test_an_unknown_break_defaults_to_joining():
    """A soft hyphen is much the commoner reason for a hyphen at a line end."""
    assert _dehyphenate("perturba- tions", set()) == "perturbations"


def test_ordinary_hyphens_are_untouched():
    # No space after the hyphen, so this never sat at a line break.
    vocab = {"single-cell", "single", "cell"}
    assert _dehyphenate("single-cell analysis", vocab) == "single-cell analysis"


def test_vocabulary_excludes_broken_forms():
    """A broken word must not vote for its own broken form."""
    lines = [
        _Line(text="the expres-", page=1, size=10.0, bold=False, top=1.0),
        _Line(text="sion of genes", page=1, size=10.0, bold=False, top=2.0),
        _Line(text="gene expression levels", page=1, size=10.0, bold=False, top=3.0),
    ]
    vocab = _document_vocabulary(lines)
    assert "expression" in vocab
    assert "expres" not in vocab


def test_end_to_end_on_a_real_string():
    lines = [
        _Line(text="models predict gene expression changes caused by",
              page=1, size=10.0, bold=False, top=1.0),
        _Line(text="perturbations of single-cell control populations",
              page=1, size=10.0, bold=False, top=2.0),
    ]
    vocab = _document_vocabulary(lines)
    broken = "claim to predict gene expres- sion caused by per- turbations of single- cell con- trol"
    assert _dehyphenate(broken, vocab) == (
        "claim to predict gene expression caused by perturbations of single-cell control"
    )


# --- title plausibility ------------------------------------------------------
# Each string here was produced by the parser on a real paper in the corpus.

from researchlens.ingest.parse import _plausible_title  # noqa: E402


def test_a_real_title_is_accepted():
    assert _plausible_title("Auto-Encoding Variational Bayes")
    assert _plausible_title("Mixtral of Experts")


def test_a_drop_cap_is_rejected():
    """The original failure: the largest glyph on page 1 is a decorative
    initial, which returned "M" for 19 of 101 papers."""
    assert not _plausible_title("M")
    assert not _plausible_title("1 3")


def test_a_pmc_cover_banner_is_rejected():
    assert not _plausible_title("HHS Public Access")
    assert not _plausible_title("Author Manuscript")


def test_letter_spaced_display_type_is_rejected():
    """Plausible by length and word count, meaningless as a title."""
    assert not _plausible_title("d e e e e e p o n n p e")
    assert not _plausible_title("V D C N L -S I R")


def test_a_section_heading_is_rejected():
    assert not _plausible_title("1 Introduction X")
