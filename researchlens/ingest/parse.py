"""Structure-preserving PDF parsing for scientific papers.

The promise this module makes to everything downstream is that a passage knows
which section it came from and which page it was printed on. Treating a PDF as
a flat string is cheaper and loses exactly the information a citation needs.

Headings are found by font geometry, not by pattern-matching prose. A paper's
body text sits at one dominant size; headings are set larger, bolder, or both.
That signal is present in essentially every typeset paper and survives the
two-column layouts and running headers that defeat regex-only approaches.

`pdfplumber` was chosen over PyMuPDF deliberately: PyMuPDF is AGPL, which would
make this repository AGPL too, and the point of publishing it is that people can
use it. pdfplumber is MIT and exposes per-character `size` and `fontname`, which
is all the geometry this needs.

Upgrade path, deliberately not taken yet: GROBID parses scholarly PDFs into TEI
with real section trees and resolved references, and is strictly better at this.
It is also a Java service, a second container, and a much slower ingest. The
seam is `parse_pdf` — swapping the implementation behind it changes nothing
above. Revisit if heading recall on the eval corpus turns out to be the thing
capping Recall@5.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from researchlens.types import Document, Section, SectionKind

# pdfminer warns per-glyph about malformed font descriptors ("Could not get
# FontBBox..."). Real papers trip this constantly and there is nothing to act
# on — the text still extracts. Left at default it emits thousands of lines per
# ingest and buries the messages that do matter.
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Headings as they are actually printed, mapped to the normalised vocabulary.
# Order matters: the first pattern that matches wins, so "related work" is
# tested before the bare "work" that no pattern claims anyway.
_SECTION_PATTERNS: list[tuple[str, SectionKind]] = [
    (r"^abstract$", "abstract"),
    (r"^(1\.?\s*)?introduction$", "introduction"),
    (r"^(related\s+work|background|prior\s+work|literature\s+review)$", "related"),
    (r"^(materials?\s+and\s+methods?|methods?|methodology|approach|"
     r"experimental\s+setup|model|architecture|data\s+and\s+methods?)$", "methods"),
    (r"^(results?|experiments?|experiments?\s+and\s+results?|evaluation|findings)$", "results"),
    (r"^(discussion|analysis|ablations?|ablation\s+stud(y|ies))$", "discussion"),
    (r"^(conclusions?|concluding\s+remarks|summary|"
     r"conclusions?\s+and\s+future\s+work)$", "conclusion"),
    (r"^(references|bibliography)$", "references"),
    (r"^(appendix|appendices|supplementary(\s+material)?)", "appendix"),
]

# Leading numbering as papers print it: "3", "3.", "3.2", "IV.", "A.1".
_NUMBERING = re.compile(r"^\s*(\d+(\.\d+)*|[IVXLC]+|[A-Z])[.)]?\s+")

# A heading is short. Sentences of body text that happen to be set large — a
# pull quote, a one-line caption — are excluded by word count more reliably
# than by anything else.
_MAX_HEADING_WORDS = 12

# Journal furniture, set in the same large or bold type as real headings.
# Running heads ("Article", "OPEN ACCESS"), figure callouts and licence lines
# are typographically indistinguishable from a section heading and, left in,
# fragment a paper into dozens of one-paragraph sections whose passages then
# inherit the wrong parent.
_FURNITURE = re.compile(
    r"^\s*(open\s*access|article|resource|review|correspondence|perspective|"
    r"brief\s+communication|letter|editorial|preprint|"
    r"(extended\s+data\s+)?(fig(ure)?|table|supplementary|supp)\b|"
    r"see\s+also|in\s+brief|highlights|graphical\s+abstract|"
    r"https?://|www\.|doi:|©|creative\s+commons)",
    re.I,
)

# A superscript citation marker: digits, or a small run of them, set well below
# body size. These are interleaved into the text by extraction — "these 9 10
# models", and worse "interfer-" "13" "ence", which breaks a word in half — so
# they cost both noise and recall.
_MARKER = re.compile(r"^[0-9]{1,3}(?:[,\u2013-][0-9]{1,3})*$")

# Fraction of a page's median word size below which a purely numeric token is a
# superscript rather than a value. Deliberately well under 1.0: metric values in
# prose ("a Dice score of 0.94") are set at body size and must survive, because
# exact numbers are precisely what lexical retrieval exists to find.
_MARKER_SIZE_RATIO = 0.82

# The opening of a figure or table caption. Journals punctuate these in every
# conceivable way — "Fig. 1 |", "Figure 2.", "Table 3:", "Extended Data Fig. 4"
# — so the separator is optional and the number is what anchors it.
_CAPTION_START = re.compile(
    r"^\s*(extended\s+data\s+)?(fig(ure)?\.?|table|supplementary\s+(fig(ure)?|table))"
    r"\s*\.?\s*\d{1,3}\b",
    re.I,
)

# Figure panel labels: "a", "a b", "f g h i". Every token a single letter.
_PANEL_LABELS = re.compile(r"^\s*[a-z]([\s,]+[a-z])*\s*$", re.I)

# Horizontal gap, in points, below which two words are treated as one word.
# pdfplumber defaults to 3, which merges words in the tight typography used by
# Cell Press and several other journals: whole abstracts extract as
# "Weidentifyover1millioncandidate", which no query can match by any means,
# lexical or dense. Verified across this corpus — 3 merges, 2 does not, and
# lowering it further changes nothing. (pdfplumber 0.11 has no
# x_tolerance_ratio, so this cannot scale with font size.)
_X_TOLERANCE = 2.0

# Horizontal gap, in points, that separates two independent runs of text on the
# same visual band — a column gutter or the space beside a margin box. Well
# above any inter-word space at body size, well below a real gutter.
_GUTTER_GAP = 18.0

# Coverage at the gutter, as a fraction of the page's peak coverage, below
# which a page is treated as two-column. Measured on this corpus: two-column
# pages trough at 0.28, single-column at 0.56 and above.
_GUTTER_RATIO = 0.40

# Below this many characters per page, the file has no usable text layer.
# See the density check in `parse_pdf` for why this is not `if not lines`.
_MIN_CHARS_PER_PAGE = 100


@dataclass(frozen=True, slots=True)
class _Line:
    """One visual line of text, with the geometry needed to judge it."""

    text: str
    page: int
    size: float
    bold: bool
    #: Vertical position, used only for ordering within a page.
    top: float


def _is_bold(fontname: str) -> bool:
    f = fontname.lower()
    return "bold" in f or "black" in f or "heavy" in f or f.endswith("-b")


def _split_on_gutters(words: list[dict]) -> list[list[dict]]:
    """Split one horizontal band of words wherever a wide gap separates them.

    Words sharing a `top` are not necessarily on the same printed line. A
    two-column paper puts two unrelated sentences at the same height, and a
    review article's margin box sits level with the body text beside it.
    Joining them produces "descriptive atlasing Introduction towards inferring"
    — a sentence that exists in no column and matches no query.

    The threshold is well above any inter-word space at body size and well
    below a column gutter, so ordinary prose is never split.
    """
    if not words:
        return []
    words = sorted(words, key=lambda w: w["x0"])
    groups: list[list[dict]] = [[words[0]]]
    for w in words[1:]:
        if w["x0"] - groups[-1][-1]["x1"] > _GUTTER_GAP:
            groups.append([w])
        else:
            groups[-1].append(w)
    return groups


def _column_split(words: list[dict], width: float) -> float | None:
    """Find the page's column gutter from where no word is printed.

    The signal is horizontal coverage computed over *raw words*. A word is a
    contiguous run of glyphs, so no word ever crosses a gutter, and a
    two-column page therefore shows a narrow band of near-zero coverage down
    its middle. A single-column page does not.

    Two earlier attempts failed on real papers and are worth recording so they
    are not tried again:

    1. Counting fragments that straddle a candidate line. Any page with a
       full-width figure — which is most pages — looked single-column.
    2. Coverage computed over merged line-fragments rather than words. The
       merged fragments already spanned the gutter, which is the very defect
       being detected, so the measurement destroyed its own signal.

    Splitting on a fixed horizontal gap cannot replace this either: justified
    body text on these papers reaches 15pt between words while a column gutter
    can be as narrow as 15pt, so the two ranges overlap and no absolute
    threshold separates them. Columns are a property of the page, not of a line.

    Measured on this corpus: two-column pages trough at 0.28 of peak coverage,
    single-column pages at 0.56 and above.
    """
    if len(words) < 40:
        return None

    bins = 80
    scale = bins / width
    coverage = [0] * bins
    for w in words:
        lo = max(0, min(bins - 1, int(w["x0"] * scale)))
        hi = max(0, min(bins - 1, int(w["x1"] * scale)))
        for i in range(lo, hi + 1):
            coverage[i] += 1

    peak = max(coverage)
    if peak == 0:
        return None

    # Only the middle can hold a gutter: a gap near an edge is a margin.
    lo_b, hi_b = int(bins * 0.32), int(bins * 0.68)
    window = coverage[lo_b:hi_b]
    if not window:
        return None

    trough = min(window)
    if trough > peak * _GUTTER_RATIO:
        return None

    idx = lo_b + window.index(trough)
    split = (idx + 0.5) / scale

    left = sum(1 for w in words if w["x1"] <= split)
    right = sum(1 for w in words if w["x0"] >= split)
    if min(left, right) < len(words) * 0.15:
        return None
    return split


def _order_fragments(
    fragments: list[tuple[float, float, float, list[dict]]], split: float | None
) -> list[tuple[float, float, float, list[dict]]]:
    """Put fragments into reading order.

    With no split, top-to-bottom then left-to-right. With one, the page is a
    sequence of bands separated by full-width elements; within each band the
    left region is read in full before the right. That is the order a person
    reads it in, and therefore the order in which the sentences are whole.
    """
    fragments = sorted(fragments, key=lambda f: (f[2], f[0]))
    if split is None:
        return fragments

    out: list[tuple[float, float, float, list[dict]]] = []
    left: list = []
    right: list = []

    def flush() -> None:
        out.extend(left)
        out.extend(right)
        left.clear()
        right.clear()

    for frag in fragments:
        x0, x1, _top, _ws = frag
        if x1 <= split:
            left.append(frag)
        elif x0 >= split:
            right.append(frag)
        else:
            # Spans both regions: it closes the band above and starts a new one.
            flush()
            out.append(frag)
    flush()
    return out


def _lines_from_page(page: pdfplumber.page.Page, page_no: int) -> list[_Line]:
    """Group words into visual lines, respecting columns and margin boxes.

    pdfplumber returns words, not lines. Words on the same printed line share a
    `top` within rounding, so they are bucketed on a rounded `top` — the
    tolerance is deliberately coarse (0.5pt) because subscripts and inline maths
    sit fractionally off the baseline and should not start a new line.

    Bucketing on `top` alone is not enough, because a page is not one column of
    text. Bands are split at wide horizontal gaps, and where the page divides
    cleanly in two, the left region is emitted in full before the right — which
    is the order a person reads it in, and therefore the order in which its
    sentences are whole.
    """
    try:
        words = page.extract_words(
            extra_attrs=["size", "fontname"],
            use_text_flow=False,
            x_tolerance=_X_TOLERANCE,
        )
    except Exception:
        # A page that is a scanned image, or otherwise has no text layer.
        return []

    # Drop superscript citation markers before anything else sees them.
    sizes = sorted(w.get("size", 0.0) for w in words if w.get("size"))
    if sizes:
        page_body = sizes[len(sizes) // 2]
        words = [
            w
            for w in words
            if not (
                _MARKER.match(w["text"])
                and w.get("size", page_body) < page_body * _MARKER_SIZE_RATIO
            )
        ]

    buckets: dict[float, list[dict]] = {}
    for w in words:
        key = round(w["top"] * 2) / 2
        buckets.setdefault(key, []).append(w)

    # Columns are decided once for the page, then applied to every line. A
    # per-line gap threshold cannot do this: justified word spacing and a
    # narrow gutter occupy the same range of gap widths.
    split = _column_split(words, float(page.width))

    # (x0, x1, top, words) for each horizontal run of words.
    fragments: list[tuple[float, float, float, list[dict]]] = []
    for top in sorted(buckets):
        bands = [buckets[top]]
        if split is not None:
            left = [w for w in buckets[top] if w["x1"] <= split]
            right = [w for w in buckets[top] if w["x0"] >= split]
            spanning = [w for w in buckets[top] if w["x0"] < split < w["x1"]]
            # A word straddling the gutter means this band is a full-width
            # element; leave it whole so `_order_fragments` treats it as a
            # separator rather than splitting a word's line in half.
            if not spanning:
                bands = [b for b in (left, right) if b]
        for band in bands:
            # Within a column, a wide gap still separates a margin box or a
            # table cell from the body beside it.
            for group in _split_on_gutters(band):
                fragments.append((group[0]["x0"], group[-1]["x1"], top, group))

    ordered = _order_fragments(fragments, split)

    lines: list[_Line] = []
    for _x0, _x1, top, ws in ordered:
        text = " ".join(w["text"] for w in ws).strip()
        if not text:
            continue
        sizes = [w.get("size", 0.0) for w in ws if w.get("size")]
        fonts = [str(w.get("fontname", "")) for w in ws]
        lines.append(
            _Line(
                text=text,
                page=page_no,
                size=round(statistics.median(sizes), 2) if sizes else 0.0,
                bold=sum(_is_bold(f) for f in fonts) > len(fonts) / 2,
                top=top,
            )
        )
    return lines


def _body_size(lines: list[_Line]) -> float:
    """The dominant text size, weighted by how much text is set in it.

    Weighting by character count rather than line count matters: a paper's
    headings and its running headers are numerous but short, and an unweighted
    mode can land on a header size instead of the body.
    """
    weights: Counter[float] = Counter()
    for ln in lines:
        weights[ln.size] += len(ln.text)
    if not weights:
        return 0.0
    return weights.most_common(1)[0][0]


def _classify(heading: str) -> SectionKind:
    """Normalise a printed heading to the shared vocabulary."""
    stripped = _NUMBERING.sub("", heading).strip().lower()
    stripped = re.sub(r"[^\w\s&]+$", "", stripped)
    stripped = stripped.replace("&", "and")
    stripped = re.sub(r"\s+", " ", stripped)
    for pattern, kind in _SECTION_PATTERNS:
        if re.match(pattern, stripped):
            return kind
    return "other"


def _looks_like_heading(line: _Line, body: float) -> bool:
    """Judge one line on geometry, then sanity-check it on shape."""
    words = line.text.split()
    if not (1 <= len(words) <= _MAX_HEADING_WORDS):
        return False
    # A line ending in a full stop is a sentence, whatever size it is set in.
    if line.text.rstrip().endswith((".", ",", ";", ":")) and not _NUMBERING.match(line.text):
        return False
    # Page numbers, running heads, stray figure labels.
    if re.fullmatch(r"[\d\s.\-–—]+", line.text):
        return False
    # Too short to name a division; usually a dropped cap or a panel letter.
    if len(line.text.strip()) <= 2:
        return False
    if _PANEL_LABELS.match(line.text) or _FURNITURE.match(line.text):
        return False

    larger = line.size >= body + 0.6
    emphasised = line.bold and line.size >= body - 0.2
    numbered = bool(_NUMBERING.match(line.text)) and (line.bold or larger)
    named = _classify(line.text) != "other" and (line.bold or larger or line.text.isupper())
    return larger or emphasised or numbered or named


# Cover-page banners that are set larger than the title on PMC and several
# journal templates, and so win a naive "largest text" rule.
_TITLE_BANNER = re.compile(
    r"^(hhs public access|nih public access|author manuscript|europe pmc|"
    r"published in final edited form|open access|research article|"
    r"original (research )?article|review article|brief communication|"
    r"downloaded from|see discussions, stats|preprint|"
    r"\d+\.?\s+(introduction|abstract|background|related work)\b)",
    re.I,
)


def _plausible_title(text: str) -> bool:
    """Whether a candidate string could be a paper's title.

    Guards against the failure this replaced: taking every line at the largest
    size on page 1 returned "T", "M", "w" and "1 3" for 19 of 101 papers,
    because the biggest glyph on the page is often a decorative drop cap or a
    journal logo rather than the title. A title shown in a citation is the
    first thing a reader checks, so a wrong one is worse than most defects.
    """
    words = text.split()
    if len(words) < 2 or len(text) < 15:
        return False
    if _TITLE_BANNER.match(text.strip()):
        return False
    # Letter-spaced display type extracts as "d e e e e e p o n n p e" and
    # "V D C N L -S I R" — plausible by every other measure, and meaningless.
    singles = sum(1 for w in words if len(w) == 1)
    if singles > len(words) * 0.5:
        return False
    letters = sum(c.isalpha() for c in text)
    # A running head of page numbers and rule characters is not a title.
    return letters >= len(text) * 0.5


def _extract_title(lines: list[_Line]) -> str:
    """The largest text on page 1 that could actually be a title.

    Size tiers are walked from largest down, and the first tier whose lines
    form something title-shaped wins. Titles wrap, so every line in the
    winning tier is joined rather than only the first.
    """
    first = [ln for ln in lines if ln.page == 1 and ln.text.strip()]
    if not first:
        return "Untitled"

    for size in sorted({round(ln.size, 1) for ln in first}, reverse=True):
        parts = [
            ln.text
            for ln in first
            if abs(round(ln.size, 1) - size) < 0.15
            and not re.match(r"^arxiv:", ln.text, re.I)
            and not _FURNITURE.match(ln.text)
        ]
        candidate = re.sub(r"\s+", " ", " ".join(parts)).strip()
        if _plausible_title(candidate):
            # A tier occasionally sweeps in a whole abstract; a title does not
            # run to 300 characters, so keep the leading sentence-ish part.
            return candidate[:300].strip()

    return "Untitled"


def _extract_authors(lines: list[_Line], title: str, body: float) -> list[str]:
    """Best-effort author extraction.

    This is genuinely hard — authors appear in every conceivable arrangement,
    interleaved with superscript affiliation markers and email addresses — and
    it is not on the critical path, because nothing in retrieval or citation
    depends on it. A wrong author list is a cosmetic defect in the source
    browser, so a heuristic that fails visibly is preferable to a dependency.

    Rule: the lines between the title and the abstract, at or slightly above
    body size, split on the separators authors are actually listed with.
    """
    first = [ln for ln in lines if ln.page == 1]
    started = False
    collected: list[str] = []
    for ln in first:
        if not started:
            if ln.text and ln.text in title:
                started = True
            continue
        if _classify(ln.text) == "abstract" or len(collected) >= 4:
            break
        if ln.size < body - 1.0 or "@" in ln.text:
            continue
        collected.append(ln.text)

    blob = " ".join(collected)
    blob = re.sub(r"[\d\*†‡§¶]+", "", blob)          # affiliation markers
    blob = re.sub(r"\b(and)\b", ",", blob, flags=re.I)
    names = [n.strip(" ,.;") for n in blob.split(",")]
    return [n for n in names if 2 <= len(n.split()) <= 5][:20]


def _heading_levels(headings: list[_Line]) -> dict[float, int]:
    """Map each heading font size to a nesting level, largest size = level 1.

    Papers signal hierarchy typographically: a section heading is set larger
    than its subsections. That signal is already in the extracted geometry and
    was previously discarded, which is why most passages classified as "other" —
    "3.2 Training details" is a real heading with no IMRaD name of its own, and
    without its parent it says nothing about where the passage came from.

    Sizes are bucketed at 0.5pt because the same logical level can vary by a
    fraction between pages, and treating those as distinct levels would produce
    a nesting depth of nine on a paper with three.
    """
    sizes = sorted({round(h.size * 2) / 2 for h in headings}, reverse=True)
    return {s: i + 1 for i, s in enumerate(sizes)}


_LINEBREAK_HYPHEN = re.compile(r"(\w+)-\s+(\w+)")


def _dehyphenate(text: str, vocab: set[str]) -> str:
    """Rejoin words broken across a line by justification.

    Typesetting splits "expression" into "expres-" and "sion" at a line end,
    and joining lines with a space preserves the break as "expres- sion" — a
    token that matches no query, lexically or densely. About 45 of these occur
    per paper in this corpus, so leaving them costs real recall.

    The difficulty is that a hyphen at a line end is ambiguous: "single-" then
    "cell" is a real compound that must keep its hyphen, while "expres-" then
    "sion" must lose it. No rule reads both correctly from the characters
    alone, so the decision is made against the *document's own vocabulary* —
    words the paper uses elsewhere, unbroken. That needs no wordlist, no
    language assumption, and adapts to the paper's own jargon.

    The ladder, in order:

    1. A continuation is lowercase. "mod-" followed by "7" is a superscript
       reference marker, and "Gene-" followed by "GEARS" is a caption artefact;
       neither is a broken word.
    2. If the joined form appears elsewhere in the document, join it.
       "expression" does, so "expres- sion" becomes "expression".
    3. If the hyphenated form appears elsewhere, keep the hyphen.
    4. If both halves are themselves words the document uses, keep the hyphen —
       "gene" and "models" are both real, so "Gene- models" is a compound
       broken at a line end, not a broken word.

    Anything left over is joined, because a soft hyphen is much the commoner
    reason for a hyphen to sit at the end of a line.
    """

    def resolve(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        if not b[:1].islower():
            return m.group(0)
        merged = f"{a}{b}"
        hyphenated = f"{a}-{b}"
        if merged.lower() in vocab:
            return merged
        if hyphenated.lower() in vocab:
            return hyphenated
        if a.lower() in vocab and b.lower() in vocab:
            return hyphenated
        return merged

    return _LINEBREAK_HYPHEN.sub(resolve, text)


def _document_vocabulary(lines: list[_Line]) -> set[str]:
    """Words the document uses, for deciding hyphenation.

    Built from tokens that are *not* adjacent to a line-break hyphen, so a
    broken word never votes for its own broken form.
    """
    joined = " ".join(ln.text for ln in lines)
    cleaned = _LINEBREAK_HYPHEN.sub(" ", joined)
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z-]{1,}", cleaned)}


def parse_pdf(path: str | Path) -> Document:
    """Parse one PDF into a `Document` with real sections and page spans."""
    path = Path(path)
    return parse_bytes(path.read_bytes(), name=path.name, source=str(path))


def parse_bytes(raw: bytes, name: str, source: str | None = None) -> Document:
    """Parse PDF bytes into a `Document` with real sections and page spans.

    Bytes rather than a path because an upload arrives as bytes, and writing it
    to a temporary file first would give the parser two entry points that can
    drift. The document id is the sha256 of those bytes either way, so a paper
    uploaded and the same paper ingested from disk are the same document —
    which is how an upload of something already in the corpus is recognised
    rather than indexed twice.

    Raises `ValueError` when the file has no extractable text layer at all,
    which for this corpus means a scanned paper. Failing loudly is correct:
    silently indexing an empty document would put a paper in the library that
    can never be retrieved, and the eval numbers would quietly worsen with no
    visible cause.
    """
    doc_id = hashlib.sha256(raw).hexdigest()[:16]

    lines: list[_Line] = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            lines.extend(_lines_from_page(page, i))

    # A scanned paper is not text-free — it usually carries a thin layer of
    # page numbers, an OCR stamp or a library watermark. Testing `if not lines`
    # lets those through to fail later with a misleading message about
    # sections. Measure the actual text density instead: a typeset paper runs
    # 2,000-4,000 characters per page, so 100 is a floor no real paper
    # approaches and no scan clears.
    chars = sum(len(ln.text) for ln in lines)
    if chars < _MIN_CHARS_PER_PAGE * max(n_pages, 1):
        raise ValueError(
            f"{name}: {chars} characters across {n_pages} pages "
            f"({chars / max(n_pages, 1):.0f}/page) — no usable text layer. "
            "This is a scanned PDF; OCR it before ingesting, or drop it from "
            "the corpus. Indexing it would add a paper that can never be "
            "retrieved and would quietly depress Recall@K."
        )

    body = _body_size(lines)
    vocab = _document_vocabulary(lines)
    title = _extract_title(lines)
    authors = _extract_authors(lines, title, body)

    # Headings are collected first so their sizes can be ranked into levels;
    # a heading's level is only meaningful relative to the other headings in
    # the same document.
    heading_lines = [
        ln for ln in lines
        if _looks_like_heading(ln, body)
        and (ln.page > 1 or _classify(ln.text) != "other")
    ]
    levels = _heading_levels(heading_lines)

    # Walk the lines once, opening a new section at every heading.
    sections: list[Section] = []
    cur_kind: SectionKind = "title"
    cur_heading = title
    cur_page = lines[0].page
    buf: list[str] = []
    last_page = lines[0].page
    # (level, kind) of each enclosing heading, outermost first. A subsection
    # with no name of its own inherits the kind of the section containing it.
    stack: list[tuple[int, SectionKind]] = []

    def close(end_page: int) -> None:
        text = " ".join(buf).strip()
        text = re.sub(r"\s+", " ", text)
        text = _dehyphenate(text, vocab)
        # Sections shorter than a sentence are heading-detection noise: a
        # two-column split heading, a caption misread as a division.
        if len(text) >= 40:
            sections.append(
                Section(
                    kind=cur_kind,
                    heading=cur_heading,
                    page_start=cur_page,
                    page_end=end_page,
                    text=text,
                )
            )

    for ln in lines:
        # On page 1 the title block, author list and affiliations are all set
        # large or bold and would each open a spurious section. So the first
        # page only yields a heading when the text is a *recognised* division
        # name; later pages accept any line that looks like one.
        # A caption opens its own passage. It is set apart typographically —
        # larger or bolder than body text — and describes a figure that is
        # often the only place a result appears. Left in the prose it is
        # neither findable nor citable; treated as a heading it would swallow
        # the paragraphs after it.
        if _CAPTION_START.match(ln.text) and (ln.size >= body - 0.2 or ln.bold):
            close(last_page)
            buf = [ln.text]
            cur_kind = "table" if re.match(r"^\s*table", ln.text, re.I) else "figure"
            cur_heading = " ".join(ln.text.split()[:8])
            cur_page = ln.page
            last_page = ln.page
            continue

        heading = _looks_like_heading(ln, body)
        if ln.page == 1:
            heading = heading and _classify(ln.text) != "other"

        if heading:
            close(last_page)
            buf = []

            level = levels.get(round(ln.size * 2) / 2, len(levels) + 1)
            named = _classify(ln.text)

            # Leave any sections this heading closes: everything at the same
            # level or deeper.
            while stack and stack[-1][0] >= level:
                stack.pop()

            if named == "other" and stack:
                # An unnamed subsection belongs to whatever contains it, so
                # "Data preprocessing" inside Methods is still methods.
                cur_kind = stack[-1][1]
            else:
                cur_kind = named

            stack.append((level, cur_kind))
            cur_heading = _NUMBERING.sub("", ln.text).strip() or ln.text
            cur_page = ln.page
        else:
            buf.append(ln.text)
        last_page = ln.page
    close(last_page)

    if not sections:
        raise ValueError(
            f"{name}: text extracted but no sections survived. "
            "Inspect with `python -m researchlens.ingest.parse <path> --debug`."
        )

    return Document(
        doc_id=doc_id,
        title=title,
        authors=authors,
        sections=sections,
        n_pages=n_pages,
        source_path=source or name,
        arxiv_id=_sniff_arxiv_id(lines),
    )


def _sniff_arxiv_id(lines: list[_Line]) -> str | None:
    for ln in lines[:60]:
        m = re.search(r"arxiv:\s*(\d{4}\.\d{4,5})(v\d+)?", ln.text, re.I)
        if m:
            return m.group(1)
    return None


def _main() -> None:
    """Inspect a single PDF's parse. The fastest way to see why a paper's
    sections came out wrong, which during corpus building is a daily question.
    """
    import argparse

    ap = argparse.ArgumentParser(description="Parse one PDF and print its structure.")
    ap.add_argument("path")
    ap.add_argument("--debug", action="store_true", help="show every section's opening words")
    args = ap.parse_args()

    doc = parse_pdf(args.path)
    print(f"{doc.title}")
    print(f"  {', '.join(doc.authors) or '(authors not detected)'}")
    print(f"  {doc.n_pages} pages · doc_id {doc.doc_id}"
          + (f" · arXiv:{doc.arxiv_id}" if doc.arxiv_id else ""))
    print()
    for s in doc.sections:
        head = f"  [{s.kind:<12}] p{s.pages:<7} {s.heading[:56]}"
        print(head)
        if args.debug:
            print(f"      {s.text[:160]}...")


if __name__ == "__main__":
    _main()
