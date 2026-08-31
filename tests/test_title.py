"""Title extraction against the three typographic traps in the real corpus.

A title is the one field a reader sees on every citation, so a wrong one is
worse than most defects — and all three of these were wrong silently, with no
exception raised and no test failing. Nineteen of a hundred and one titles were
broken before these fixtures existed.
"""

from __future__ import annotations

from researchlens.ingest.parse import _display_size, parse_bytes
from tests.pdfs import (
    make_drop_cap_pdf,
    make_pdf,
    make_rotated_margin_pdf,
    make_small_caps_pdf,
)


def test_small_caps_keep_their_initials():
    """ICLR and NeurIPS set a title's first letters at full size and the rest
    at small-cap size, on one baseline. Bucketing lines by their top split the
    two, and the tier of beheaded remainders won: "AN IMAGE IS WORTH 16X16
    WORDS" extracted as "N MAGE IS ORTH X ORDS"."""
    raw = make_small_caps_pdf(
        [("A", "N"), ("I", "MAGE"), ("I", "S"), ("W", "ORTH"), ("W", "ORDS")]
    )
    assert parse_bytes(raw, "vit.pdf").title == "AN IMAGE IS WORTH WORDS"


def test_a_drop_cap_does_not_promote_its_body_line():
    """One 29.4pt initial on a 10pt line is not display type. Tiering on the
    line's maximum said it was, and the title became "is introduced to
    effectively balance the dual objectives U"."""
    title = parse_bytes(make_drop_cap_pdf(), "uscnet.pdf").title
    assert title.startswith("Urolithiasis Classification")


def test_a_rotated_margin_stamp_is_not_part_of_the_title():
    """arXiv prints its identifier sideways up the left edge. Its glyphs share
    an x and spread over many tops, so they scattered into nearby lines and
    seven titles ended in "A V X"."""
    title = parse_bytes(make_rotated_margin_pdf(), "swin.pdf").title
    assert title == "Swin Transformer Hierarchical Vision Using Shifted Windows"
    assert "arXiv" not in title and "2103" not in title


def test_an_ordinary_title_is_unaffected():
    raw = make_pdf(
        ["A Study Of Widget Calibration In Practice"]
        + ["Body sentence about widget calibration and its effects."] * 12,
        [16.0] + [11.0] * 12,
    )
    assert parse_bytes(raw, "w.pdf").title == "A Study Of Widget Calibration In Practice"


# ---- the quantile itself ---------------------------------------------------

def test_display_size_ignores_a_lone_giant():
    # A drop cap: one glyph in ten.
    assert _display_size([29.4] + [10.0] * 9) == 10.0


def test_display_size_follows_small_caps():
    # Both lines of a small-caps title, whose medians differ.
    assert _display_size([17.22] * 7 + [13.77] * 5) == 17.22
    assert _display_size([17.22] * 4 + [13.77] * 6) == 17.22


def test_display_size_of_uniform_text_is_that_size():
    assert _display_size([11.0] * 20) == 11.0


def test_display_size_of_nothing_is_zero():
    assert _display_size([]) == 0.0


# ---- journal mastheads -----------------------------------------------------

def test_a_journal_masthead_is_not_a_title():
    """Elsevier sets the journal's name larger than the paper's — 13.9pt over
    13.4 — so half a point put "Alexandria Engineering Journal" on every
    citation of that paper."""
    from researchlens.ingest.parse import _plausible_title

    assert not _plausible_title("Alexandria Engineering Journal")
    assert not _plausible_title("Results in Engineering")


def test_a_real_title_containing_a_venue_word_survives():
    """The guard is bounded by length, because a paper in this corpus is
    called "A Review of Diabetic Retinopathy Datasets…" and another is called
    "Mixtral of Experts"."""
    from researchlens.ingest.parse import _plausible_title

    assert _plausible_title(
        "A Review of Diabetic Retinopathy Datasets, Approaches, "
        "Evaluation Metrics and Future Trends"
    )
    assert _plausible_title("Mixtral of Experts")
    assert _plausible_title(
        "Journal of Machine Learning Applied to Renal Imaging Studies"
    )
