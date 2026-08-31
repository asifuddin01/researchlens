"""Live literature search against arXiv.

Benchmark B asks what a field is doing *now* — current trends, recent work,
what is state of the art. A fixed corpus cannot answer that however good its
retrieval is, and the failure mode is the dangerous one: the system answered
"current research trends in large language models" from Attention Is All You
Need (2017) and BERT (2018), with real citations and no sign the evidence was
eight years old. Technically grounded, substantively misleading.

This closes that gap by fetching papers published in a date window and
grounding the answer in *their* abstracts.

Two limits are structural and are surfaced rather than hidden:

- **Abstracts, not full text.** arXiv's API returns metadata. An abstract
  supports "this paper claims X"; it does not support "this paper measured X
  on dataset Y", because the abstract may not say and the reader cannot check.
  Live evidence is therefore labelled as abstract-level everywhere it appears.
- **arXiv is not the literature.** It skews to ML, physics and quantitative
  biology, and misses journal-only work entirely — most clinical and much
  biological research. A question about radiology practice will find less here
  than the field contains, and an answer must not read as a survey.
"""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, timedelta

import httpx

from researchlens.types import Chunk

API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}

#: arXiv asks automated clients to leave a gap between requests. One search per
#: question is well inside that; the delay is here so a batch stays polite.
_MIN_INTERVAL = 3.0
_last_call = 0.0


@dataclass(frozen=True, slots=True)
class LivePaper:
    """A paper found by live search. Abstract only, never full text."""

    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    #: ISO date, so recency is checkable rather than asserted.
    published: str
    url: str
    source: str = "arxiv"

    @property
    def year(self) -> str:
        return self.published[:4]


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def build_query(question: str, max_terms: int = 6) -> str:
    """Turn a question into an arXiv query.

    Deliberately crude. arXiv's search is lexical, so the useful signal is the
    content words; adding more only narrows the result set until it is empty.
    Question scaffolding ("what are the major current research trends in…")
    matches nothing and is dropped.
    """
    stop = {
        "what", "are", "the", "major", "current", "recent", "research", "trends",
        "trend", "in", "of", "for", "and", "or", "a", "an", "is", "on", "to",
        "which", "how", "why", "does", "do", "open", "problems", "problem",
        "directions", "direction", "gaps", "gap", "main", "most", "important",
        "unresolved", "emerging", "latest", "state", "art", "field", "recently",
    }
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z-]{2,}", question.lower())
             if w not in stop]
    terms = words[:max_terms] or words[:3] or ["machine learning"]
    return " AND ".join(f'all:"{t}"' for t in terms)


async def search(
    question: str,
    max_results: int = 8,
    since_days: int | None = 730,
    timeout: float = 20.0,
) -> list[LivePaper]:
    """Fetch recent arXiv papers matching a question.

    Sorted by submission date rather than relevance: the question being asked
    is what is *recent*, and arXiv's relevance ordering happily returns a
    seminal 2017 paper first, which is the exact failure this exists to fix.
    """
    global _last_call

    now = asyncio.get_event_loop().time()
    if (wait := _MIN_INTERVAL - (now - _last_call)) > 0:
        await asyncio.sleep(wait)
    _last_call = asyncio.get_event_loop().time()

    params = {
        "search_query": build_query(question),
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(
            API,
            params=params,
            headers={"User-Agent": "ResearchLens/0.1 (research assistant; github.com/asifuddin01)"},
        )
        r.raise_for_status()
        root = ET.fromstring(r.text)

    cutoff = (date.today() - timedelta(days=since_days)).isoformat() if since_days else None

    papers: list[LivePaper] = []
    for entry in root.findall("atom:entry", NS):
        published = _clean(entry.findtext("atom:published", default="", namespaces=NS))[:10]
        if cutoff and published and published < cutoff:
            continue
        raw_id = _clean(entry.findtext("atom:id", default="", namespaces=NS))
        paper_id = raw_id.rsplit("/", 1)[-1]
        papers.append(
            LivePaper(
                paper_id=paper_id,
                title=_clean(entry.findtext("atom:title", default="", namespaces=NS)),
                authors=[
                    _clean(a.findtext("atom:name", default="", namespaces=NS))
                    for a in entry.findall("atom:author", NS)
                ][:8],
                abstract=_clean(entry.findtext("atom:summary", default="", namespaces=NS)),
                published=published,
                url=raw_id,
            )
        )
    return papers


#: Live evidence is marked in the chunk id itself. Everything downstream —
#: citation rendering, the evidence panel, the API — can then tell corpus
#: evidence from live evidence without a parallel code path, and a reader is
#: never shown an abstract as though it were a passage from a full paper.
#:
#: Each source uses its own prefix. An earlier version hardcoded "arxiv", so a
#: PubMed record was cited as "arxiv:PMID39541441" — a citation lying about
#: where it came from, which is worse than an uncited claim because it looks
#: checkable.
LIVE_PREFIXES = ("arxiv", "pubmed")


def is_live(chunk_id: str) -> bool:
    return any(chunk_id.startswith(f"{p}:") for p in LIVE_PREFIXES)


def to_chunks(papers: list[LivePaper]) -> list[Chunk]:
    """Adapt live results into the shape the rest of the system already speaks.

    Reusing `Chunk` rather than introducing a second evidence type means
    fusion, prompting and citation resolution work unchanged. The honesty is
    carried in the fields: `pages` says "abstract" because there is no page,
    and the heading names the source and the date, so a citation reads
    "arXiv 2608.28476 · abstract · 2026-08-28" and cannot be mistaken for a
    passage someone could turn to.
    """
    out: list[Chunk] = []
    for i, p in enumerate(papers):
        out.append(
            Chunk(
                chunk_id=f"{p.source}:{p.paper_id}",
                doc_id=f"{p.source}:{p.paper_id}",
                ordinal=i,
                text=p.abstract,
                section_kind="abstract",
                section_heading=(
                    f"{'arXiv' if p.source == 'arxiv' else 'PubMed'} {p.paper_id}"
                    f" · abstract · {p.published}"
                ),
                page_start=0,
                page_end=0,
                doc_title=p.title,
            )
        )
    return out
