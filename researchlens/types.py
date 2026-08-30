"""Core record types.

These exist in their own module because everything else depends on them and
nothing here depends on anything else. The shapes are deliberately settled
before any retrieval code is written: `Chunk` carries `section` and the page
span from the moment it is created, so a citation can name a place in a paper
without a second pass over the PDF. Retrofitting that later would invalidate
every hand-written label in `eval/questions.jsonl`, which is the one cost this
project cannot absorb.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# A paper's own division, normalised. Parsers see dozens of spellings
# ("2. MATERIALS AND METHODS", "Methods", "Materials & Methods"); retrieval and
# citation display only ever see one of these.
SectionKind = Literal[
    "title",
    "abstract",
    "introduction",
    "related",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "references",
    "appendix",
    "other",
]


@dataclass(frozen=True, slots=True)
class Section:
    """One division of a paper, as printed."""

    kind: SectionKind
    #: The heading exactly as it appeared, for display. "3.2 Training details"
    heading: str
    #: 1-indexed, inclusive, matching what a reader sees on the page.
    page_start: int
    page_end: int
    text: str

    @property
    def pages(self) -> str:
        """Human-readable page span: "7" or "7-9"."""
        if self.page_start == self.page_end:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"


@dataclass(frozen=True, slots=True)
class Document:
    """One parsed paper."""

    #: Stable across re-ingests: sha256 of the file bytes, first 16 hex chars.
    #: Used for deduplication, so the same paper added twice is added once.
    doc_id: str
    title: str
    authors: list[str]
    sections: list[Section]
    n_pages: int
    source_path: str
    #: Populated only when the paper came from a known index.
    arxiv_id: str | None = None
    doi: str | None = None

    @property
    def abstract(self) -> str:
        for s in self.sections:
            if s.kind == "abstract":
                return s.text
        return ""


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable passage, carrying enough context to cite itself.

    `chunk_id` is deterministic — `{doc_id}:{ordinal}` — so a ground-truth label
    written by hand today still resolves after the index is rebuilt, provided
    the chunking parameters have not changed. When they do change, the eval
    harness says so loudly rather than silently scoring against stale ids.
    """

    chunk_id: str
    doc_id: str
    #: Position within the document, 0-indexed.
    ordinal: int
    text: str
    section_kind: SectionKind
    section_heading: str
    page_start: int
    page_end: int
    #: Denormalised for display; a citation should not need a second lookup.
    doc_title: str

    @property
    def pages(self) -> str:
        if self.page_start == self.page_end:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"


@dataclass(frozen=True, slots=True)
class Retrieved:
    """A chunk with the scores that selected it.

    Every stage that touches a candidate records its own score rather than
    overwriting one field, because the ablation needs to see what each stage
    contributed. A candidate that BM25 found and dense retrieval missed is a
    fact worth keeping.
    """

    chunk: Chunk
    #: Whichever score ordered this result at the point it was returned.
    score: float
    bm25_score: float | None = None
    dense_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    #: Which retrievers proposed this candidate at all.
    sources: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class Citation:
    """A marker in an answer, bound to the passage that supports it."""

    #: 1-indexed, matching the "[1]" printed in the answer.
    marker: int
    chunk_id: str
    doc_id: str
    doc_title: str
    section_heading: str
    pages: str
    #: The exact passage, for the evidence panel.
    quote: str


@dataclass(frozen=True, slots=True)
class Answer:
    """A generated answer and everything needed to audit it."""

    question: str
    text: str
    citations: list[Citation]
    #: Which provider and model produced `text`, for the two-model comparison.
    model: str
    provider: str
    #: Milliseconds, split so the demo can show where the time went.
    retrieval_ms: float
    generation_ms: float
