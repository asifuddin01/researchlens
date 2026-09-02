"""Live search against PubMed.

arXiv is not the literature. It skews to machine learning, physics and
quantitative biology, and misses journal-only work entirely — which is most
clinical research and much of biology. For a corpus half of which is single-cell
genomics and medical imaging, that is the larger half of the field missing.

PubMed covers it: 37 million citations across MEDLINE and the life-science
journals, free, and with no API key. NCBI asks that automated clients identify
themselves and stay under three requests a second, both of which are honoured
below.

Two requests per search, because E-utilities separates them: `esearch` returns
identifiers, `efetch` returns records. The alternative is `esummary`, which is
one request but carries no abstract — and an abstract is the whole point, since
a title supports nothing.
"""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import httpx

from researchlens.live import query
from researchlens.live.arxiv import LivePaper

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

#: NCBI asks unauthenticated clients to stay under three requests per second.
#: A search costs two, so this spacing keeps a burst inside the limit.
_MIN_INTERVAL = 0.4
_last_call = 0.0

#: NCBI asks automated clients to identify themselves so they can be contacted
#: rather than simply blocked.
_TOOL = "ResearchLens"
_EMAIL = "researchlens@users.noreply.github.com"

def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def build_query(question: str, max_terms: int = 5) -> str:
    """Turn a question into a PubMed query.

    Content words come from `researchlens.live.query`, which keeps two-letter
    tokens: requiring three characters silently dropped "AI" from "the major
    open problems in AI for radiology", leaving "radiology" alone — which,
    sorted by date, returned tantalum implants. A dropped acronym is not a
    small loss when the acronym is the subject.

    Terms are ANDed, so an expansion is a parenthesised OR rather than another
    term, for the same reason as arXiv: demanding both forms is narrower than
    the acronym alone.
    """
    words = query.terms(question, max_terms=max_terms)
    if not words:
        # No subject. "biomedical research" used to stand in here, which
        # searched for something the reader had not asked about and returned
        # abstracts that looked like evidence.
        return ""
    return " AND ".join(
        forms[0] if len(forms) == 1 else "(" + " OR ".join(
            f'"{f}"' if " " in f else f for f in forms
        ) + ")"
        for forms in (query.expand(w) for w in words)
    )


async def _throttle() -> None:
    global _last_call
    loop = asyncio.get_event_loop()
    if (wait := _MIN_INTERVAL - (loop.time() - _last_call)) > 0:
        await asyncio.sleep(wait)
    _last_call = loop.time()


def _abstract(article: ET.Element) -> str:
    """Join a structured abstract into one passage.

    Clinical abstracts are sectioned — BACKGROUND, METHODS, RESULTS — and each
    section is its own element. Concatenating without the labels loses the
    distinction between what a study set out to do and what it found, which is
    exactly the distinction this project exists to preserve.
    """
    parts: list[str] = []
    for node in article.iter("AbstractText"):
        label = node.get("Label")
        text = _clean("".join(node.itertext()))
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts)


async def search(
    question: str,
    max_results: int = 8,
    since_days: int | None = 730,
    timeout: float = 25.0,
) -> list[LivePaper]:
    """Fetch recent PubMed records matching a question."""
    term = build_query(question)
    if not term:
        return []
    if since_days:
        start = (date.today() - timedelta(days=since_days)).strftime("%Y/%m/%d")
        term = f"({term}) AND (\"{start}\"[Date - Publication] : \"3000\"[Date - Publication])"

    async with httpx.AsyncClient(timeout=timeout) as client:
        await _throttle()
        r = await client.get(
            f"{BASE}/esearch.fcgi",
            params={
                # Relevance order, with recency enforced by the date filter in
                # `term` instead. Sorting by date on a broad query returns
                # whatever was published most recently that happens to contain
                # the word — for "radiology" that is thousands of papers a day
                # and none of them an answer. arXiv is the opposite case and is
                # sorted by date there, because its relevance ranking reliably
                # surfaces the seminal old paper first.
                "db": "pubmed", "term": term, "retmax": str(max_results),
                "retmode": "json", "sort": "relevance", "tool": _TOOL, "email": _EMAIL,
            },
        )
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        await _throttle()
        r = await client.get(
            f"{BASE}/efetch.fcgi",
            params={
                "db": "pubmed", "id": ",".join(ids), "retmode": "xml",
                "tool": _TOOL, "email": _EMAIL,
            },
        )
        r.raise_for_status()
        root = ET.fromstring(r.text)

    papers: list[LivePaper] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _clean(article.findtext(".//PMID"))
        abstract = _abstract(article)
        # A record with no abstract supports nothing. A title is a claim about
        # what a paper is called, not about what it found.
        if not pmid or not abstract:
            continue
        year = _clean(article.findtext(".//PubDate/Year")) or _clean(
            article.findtext(".//ArticleDate/Year")
        )
        month = _clean(article.findtext(".//PubDate/Month")) or "01"
        month = month if month.isdigit() else "01"
        day = _clean(article.findtext(".//PubDate/Day")) or "01"
        published = f"{year or '????'}-{month.zfill(2)}-{day.zfill(2)}"

        authors = []
        for a in article.findall(".//Author")[:8]:
            last, initials = _clean(a.findtext("LastName")), _clean(a.findtext("Initials"))
            if last:
                authors.append(f"{initials} {last}".strip())

        papers.append(
            LivePaper(
                paper_id=f"PMID{pmid}",
                title=_clean(article.findtext(".//ArticleTitle")),
                authors=authors,
                abstract=abstract,
                published=published,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                source="pubmed",
            )
        )
    return papers
