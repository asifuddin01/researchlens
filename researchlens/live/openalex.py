"""Live search against OpenAlex.

arXiv and PubMed leave a hole that is easy to describe: arXiv is preprints,
mostly machine learning and physics; PubMed is the life sciences. Between them
they miss the engineering and computer-science *journal* literature almost
entirely — IEEE, ACM, Elsevier, Springer — which for a question about, say,
federated learning in medical imaging is where much of the answer lives.

OpenAlex indexes it: ~250 million works with metadata and, for a large share,
abstracts. Free, no key, no subscription. It is the honest answer to "why
doesn't this search IEEE Xplore" — Xplore's own API needs an institutional
key, so it cannot be a default here, but its records are in OpenAlex and can
be.

Two things worth knowing about the data.

**Abstracts arrive inverted.** OpenAlex stores an `abstract_inverted_index` —
`{word: [positions]}` — reportedly to sidestep the copyright question of
storing abstracts as prose. Reconstructing it is exact for the words present,
but punctuation attaches to tokens rather than being positioned, so a rebuilt
abstract is very slightly rougher than the publisher's. It is the real text,
not a summary.

**Coverage is metadata-wide, abstract-narrow.** Plenty of records have no
abstract at all. Those are dropped rather than cited, for the same reason
PubMed records without one are: a title is a claim about what a paper is
called, not about what it found.
"""

from __future__ import annotations

import re

import httpx

from researchlens.live import query
from researchlens.live.arxiv import LivePaper

API = "https://api.openalex.org/works"

#: OpenAlex asks automated clients to identify themselves, and routes those
#: that do into a faster pool with a much higher rate limit. Not identifying
#: yourself is both ruder and slower.
_MAILTO = "researchlens@users.noreply.github.com"

def build_query(question: str, max_terms: int = 8) -> str:
    """Content words, space-separated.

    OpenAlex's `search` is a ranked full-text match over title, abstract and
    fulltext, not a boolean AND. Extra terms therefore *widen and reorder*
    rather than narrow to nothing, which is the opposite of PubMed's behaviour
    — so this keeps more of them.
    """
    # Expansions are simply appended here. Where arXiv and PubMed need an OR
    # group because their terms are ANDed, a ranked match treats an extra term
    # as extra evidence for what the question is about, so "llm large language
    # model" ranks the papers that say either — and the ones that say both
    # highest, which is right.
    out: list[str] = []
    for w in query.terms(question, max_terms=max_terms):
        for form in query.expand(w):
            if form not in out:
                out.append(form)
    return " ".join(out)


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    """Rebuild prose from OpenAlex's inverted index.

    Positions can be sparse or repeated; sorting by position and joining is
    correct for both. A missing index yields "", which the caller drops.
    """
    if not inverted:
        return ""
    slots: list[tuple[int, str]] = [
        (pos, word) for word, positions in inverted.items() for pos in positions
    ]
    if not slots:
        return ""
    slots.sort(key=lambda t: t[0])
    return re.sub(r"\s+", " ", " ".join(word for _, word in slots)).strip()


def _authors(work: dict) -> list[str]:
    out = []
    for a in work.get("authorships") or []:
        name = ((a or {}).get("author") or {}).get("display_name")
        if name:
            out.append(name)
    return out[:8]


async def search(
    question: str,
    max_results: int = 8,
    since_days: int | None = 730,
    timeout: float = 20.0,
) -> list[LivePaper]:
    """Fetch recent OpenAlex records matching a question."""
    from datetime import date, timedelta

    query = build_query(question)
    if not query:
        return []

    # `has_abstract` is applied server-side so the page is not half-filled with
    # records that will be dropped here. Asking for more than needed and
    # filtering locally wastes the request that recency already narrowed.
    filters = ["has_abstract:true"]
    if since_days:
        start = (date.today() - timedelta(days=since_days)).isoformat()
        filters.append(f"from_publication_date:{start}")

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(
            API,
            params={
                "search": query,
                "filter": ",".join(filters),
                "per-page": str(max(1, min(max_results, 50))),
                "select": "id,doi,title,publication_date,authorships,"
                          "abstract_inverted_index,primary_location",
                "mailto": _MAILTO,
            },
            headers={"User-Agent": f"ResearchLens (mailto:{_MAILTO})"},
        )
        r.raise_for_status()
        results = r.json().get("results") or []

    papers: list[LivePaper] = []
    for work in results:
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
        title = re.sub(r"\s+", " ", work.get("title") or "").strip()
        if not abstract or not title:
            continue
        # The DOI is the citable identifier and the one a reader can resolve;
        # the OpenAlex id is a fallback so a record without a DOI is still
        # addressable rather than dropped.
        doi = (work.get("doi") or "").replace("https://doi.org/", "")
        oa_id = (work.get("id") or "").rsplit("/", 1)[-1]
        url = work.get("doi") or work.get("id") or ""
        venue = (
            ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
            or ""
        )
        papers.append(
            LivePaper(
                paper_id=doi or oa_id,
                title=title,
                authors=_authors(work),
                # The venue is prepended to the abstract text rather than kept
                # in a field of its own, because it is the thing that makes an
                # OpenAlex result recognisable as journal work — and a reader
                # deciding whether to trust an abstract wants to know it ran in
                # IEEE TMI rather than nowhere.
                abstract=f"[{venue}] {abstract}" if venue else abstract,
                published=work.get("publication_date") or "",
                url=url,
                source="openalex",
            )
        )
    return papers
