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


def _lines_from_page(page: pdfplumber.page.Page, page_no: int) -> list[_Line]:
    """Group words into visual lines.

    pdfplumber returns words, not lines. Words on the same printed line share a
    `top` within rounding, so they are bucketed on a rounded `top`. The
    tolerance is deliberately coarse (0.5pt) because subscripts and inline maths
    sit fractionally off the baseline and should not start a new line.
    """
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"], use_text_flow=False)
    except Exception:
        # A page that is a scanned image, or otherwise has no text layer.
        return []

    buckets: dict[float, list[dict]] = {}
    for w in words:
        key = round(w["top"] * 2) / 2
        buckets.setdefault(key, []).append(w)

    lines: list[_Line] = []
    for top in sorted(buckets):
        ws = sorted(buckets[top], key=lambda w: w["x0"])
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

    larger = line.size >= body + 0.6
    emphasised = line.bold and line.size >= body - 0.2
    numbered = bool(_NUMBERING.match(line.text)) and (line.bold or larger)
    named = _classify(line.text) != "other" and (line.bold or larger or line.text.isupper())
    return larger or emphasised or numbered or named


def _extract_title(lines: list[_Line]) -> str:
    """The title is the largest text on the first page, read in order.

    Titles wrap, so every line at the maximum size on page 1 is joined rather
    than only the first. Lines above the title (a journal banner, an arXiv
    stamp) are set smaller and drop out on their own.
    """
    first = [ln for ln in lines if ln.page == 1]
    if not first:
        return "Untitled"
    top_size = max(ln.size for ln in first)
    parts = [
        ln.text
        for ln in first
        if ln.size >= top_size - 0.1 and not re.match(r"^arxiv:", ln.text, re.I)
    ]
    title = " ".join(parts).strip()
    return re.sub(r"\s+", " ", title) or "Untitled"


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


def parse_pdf(path: str | Path) -> Document:
    """Parse one PDF into a `Document` with real sections and page spans.

    Raises `ValueError` when the file has no extractable text layer at all,
    which for this corpus means a scanned paper. Failing loudly is correct:
    silently indexing an empty document would put a paper in the library that
    can never be retrieved, and the eval numbers would quietly worsen with no
    visible cause.
    """
    path = Path(path)
    raw = path.read_bytes()
    doc_id = hashlib.sha256(raw).hexdigest()[:16]

    lines: list[_Line] = []
    with pdfplumber.open(path) as pdf:
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
            f"{path.name}: {chars} characters across {n_pages} pages "
            f"({chars / max(n_pages, 1):.0f}/page) — no usable text layer. "
            "This is a scanned PDF; OCR it before ingesting, or drop it from "
            "the corpus. Indexing it would add a paper that can never be "
            "retrieved and would quietly depress Recall@K."
        )

    body = _body_size(lines)
    title = _extract_title(lines)
    authors = _extract_authors(lines, title, body)

    # Walk the lines once, opening a new section at every heading.
    sections: list[Section] = []
    cur_kind: SectionKind = "title"
    cur_heading = title
    cur_page = lines[0].page
    buf: list[str] = []
    last_page = lines[0].page

    def close(end_page: int) -> None:
        text = " ".join(buf).strip()
        text = re.sub(r"\s+", " ", text)
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
        heading = _looks_like_heading(ln, body)
        if ln.page == 1:
            heading = heading and _classify(ln.text) != "other"

        if heading:
            close(last_page)
            buf = []
            cur_kind = _classify(ln.text)
            cur_heading = _NUMBERING.sub("", ln.text).strip() or ln.text
            cur_page = ln.page
        else:
            buf.append(ln.text)
        last_page = ln.page
    close(last_page)

    if not sections:
        raise ValueError(
            f"{path.name}: text extracted but no sections survived. "
            "Inspect with `python -m researchlens.ingest.parse <path> --debug`."
        )

    return Document(
        doc_id=doc_id,
        title=title,
        authors=authors,
        sections=sections,
        n_pages=n_pages,
        source_path=str(path),
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
