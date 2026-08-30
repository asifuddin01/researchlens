"""Page-layout tests.

These cover the geometry that decides whether a sentence survives extraction.
Every case here is drawn from a failure observed on a real paper in the corpus,
not invented: merged columns, a margin table-of-contents beside body text, and
a full-width title above two columns.
"""

from researchlens.ingest.parse import _column_split, _order_fragments, _split_on_gutters


def _w(x0, x1, text="x"):
    return {"x0": x0, "x1": x1, "text": text, "top": 100.0}


def test_ordinary_prose_is_not_split():
    """Inter-word spaces must never be mistaken for a gutter."""
    words = [_w(50, 80), _w(84, 120), _w(124, 160)]
    assert len(_split_on_gutters(words)) == 1


def test_a_wide_gap_splits_the_band():
    """A margin box level with body text produced
    'descriptive atlasing Introduction towards inferring' before this."""
    body = [_w(40, 387, "atlasing")]
    sidebar = [_w(438, 488, "Introduction")]
    groups = _split_on_gutters(body + sidebar)
    assert len(groups) == 2
    assert groups[0][0]["text"] == "atlasing"
    assert groups[1][0]["text"] == "Introduction"


def test_split_on_gutters_handles_an_empty_band():
    assert _split_on_gutters([]) == []


def _frags(n_left, n_right, straddlers=0, width=600.0):
    out = []
    for i in range(n_left):
        out.append((40.0, 280.0, float(i * 12)))
    for i in range(n_right):
        out.append((320.0, 560.0, float(i * 12)))
    for i in range(straddlers):
        out.append((40.0, 560.0, float(-10 - i)))
    return out


def test_two_columns_are_detected():
    split = _column_split(_frags(10, 10), 600.0)
    assert split is not None
    assert 280 < split < 320


def test_a_full_width_title_does_not_defeat_detection():
    """The bug this replaced: requiring that nothing straddle the line meant a
    single title above two columns disabled column ordering for the page, and
    every real paper has one."""
    assert _column_split(_frags(10, 10, straddlers=1), 600.0) is not None


def test_single_column_is_not_split():
    frags = [(40.0, 560.0, float(i * 12)) for i in range(12)]
    assert _column_split(frags, 600.0) is None


def test_a_lone_page_number_is_not_a_column():
    assert _column_split(_frags(20, 1), 600.0) is None


def test_too_few_fragments_to_judge():
    assert _column_split(_frags(2, 2), 600.0) is None


def test_columns_are_read_left_then_right():
    left = (40.0, 280.0, 50.0, [_w(40, 280, "L1")])
    right = (320.0, 560.0, 50.0, [_w(320, 560, "R1")])
    left2 = (40.0, 280.0, 70.0, [_w(40, 280, "L2")])
    right2 = (320.0, 560.0, 70.0, [_w(320, 560, "R2")])
    ordered = _order_fragments([left, right, left2, right2], split=300.0)
    assert [f[3][0]["text"] for f in ordered] == ["L1", "L2", "R1", "R2"]


def test_a_full_width_element_separates_bands():
    """A spanning caption mid-page must close the columns above it rather than
    letting text from below flow into the column above."""
    above_l = (40.0, 280.0, 50.0, [_w(40, 280, "L1")])
    above_r = (320.0, 560.0, 50.0, [_w(320, 560, "R1")])
    caption = (40.0, 560.0, 60.0, [_w(40, 560, "CAPTION")])
    below_l = (40.0, 280.0, 70.0, [_w(40, 280, "L2")])
    below_r = (320.0, 560.0, 70.0, [_w(320, 560, "R2")])
    ordered = _order_fragments([above_l, above_r, caption, below_l, below_r], split=300.0)
    assert [f[3][0]["text"] for f in ordered] == ["L1", "R1", "CAPTION", "L2", "R2"]


def test_without_a_split_order_is_top_to_bottom():
    a = (40.0, 560.0, 70.0, [_w(40, 560, "second")])
    b = (40.0, 560.0, 50.0, [_w(40, 560, "first")])
    ordered = _order_fragments([a, b], split=None)
    assert [f[3][0]["text"] for f in ordered] == ["first", "second"]
