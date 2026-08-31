"""Live search across several literature sources.

Which sources exist here is decided by one rule: a source that needs a paid or
institutional API key cannot be a default, because "runs with no API key" is
the claim this project makes on its front page. A default that quietly fails
for anyone without a subscription is worse than an absent feature.

  arXiv     free, no key. Preprints in ML, physics, quantitative biology.
            Fast — often the same week — and misses journal-only work entirely.

  PubMed    free, no key. 37 million citations across MEDLINE and the
            life-science journals. Covers the clinical and biological
            literature arXiv does not, at the cost of appearing months later.

  Semantic  free, keyless at a low rate limit. Broad coverage across fields
  Scholar   with citation counts. Not yet wired up; the next one to add.

  IEEE      needs an API key, institutional or paid.
  Xplore

  Scopus    needs an Elsevier key and a subscription.

The last two are deliberately absent rather than forgotten. They can be added
behind configuration for anyone who has a key, in the same way the hosted
generation provider is optional — but neither can be a default.

Sources are queried concurrently, because two sequential HTTP round trips is
the difference between a question that feels answered and one that feels hung.
"""

from __future__ import annotations

import asyncio

from researchlens.live import arxiv, pubmed
from researchlens.live.arxiv import LivePaper

#: Source name -> its async search function. Adding a source is adding a row.
SOURCES = {
    "arxiv": arxiv.search,
    "pubmed": pubmed.search,
}


async def search(
    question: str,
    sources: list[str] | None = None,
    per_source: int = 5,
    since_days: int | None = 730,
) -> list[LivePaper]:
    """Query every source at once and merge the results.

    A source that fails is dropped, not raised. Live search is an enhancement:
    if arXiv is unreachable, a PubMed-only answer is better than an error, and
    an answer from the corpus alone is better than no answer at all.

    Results are interleaved rather than concatenated, so a question that both
    sources can speak to is not answered entirely from whichever was listed
    first — the prompt cites what it reads first, and ordering is therefore a
    silent editorial decision.
    """
    chosen = sources or list(SOURCES)
    tasks = [
        SOURCES[name](question, max_results=per_source, since_days=since_days)
        for name in chosen
        if name in SOURCES
    ]
    if not tasks:
        return []

    settled = await asyncio.gather(*tasks, return_exceptions=True)
    per: list[list[LivePaper]] = [r for r in settled if isinstance(r, list)]

    merged: list[LivePaper] = []
    seen: set[str] = set()
    for i in range(max((len(r) for r in per), default=0)):
        for results in per:
            if i >= len(results):
                continue
            paper = results[i]
            # The same work often appears as a preprint and as a journal
            # article. Titles are compared loosely because the two records
            # rarely punctuate them identically.
            key = "".join(ch for ch in paper.title.lower() if ch.isalnum())[:80]
            if key in seen:
                continue
            seen.add(key)
            merged.append(paper)
    return merged
