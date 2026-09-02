"""The Elementa — the author's own textbook, as a source.

A second web-fetched corpus alongside `author.py`, and deliberately not part of
it. That one answers "who wrote this"; this one answers "how does attention
work", in the author's own words, cited to a numbered proposition a reader can
open. Folding eighty-three propositions into the author corpus would bury the
profile it exists to surface.

**Fetched, and therefore current.** The site publishes `/elementa.json`, rebuilt
by `astro build` on every push to main — CMS commits included. Writing a
proposition makes it answerable within one cache window, with no redeploy here
and nothing to remember. A copy vendored into this repository would be a
snapshot that silently disagrees with the site the moment either changed.

**It competes on merit.** Propositions are reranked against the question like
everything else and given a small reserved allocation with a relevance floor,
so the textbook appears when it is the best answer and stays out when a paper
is. It is labelled as a textbook in the prompt and cited as a proposition, not
as a passage from a paper, because those are different kinds of claim: a
proposition is what the author teaches, not what an experiment measured.
"""

from __future__ import annotations

import os
import re
import time

import httpx

from researchlens.types import Chunk

CORPUS_URL = os.getenv("ELEMENTA_CORPUS_URL", "https://asifuddin.com/elementa.json")

#: Same window as the author corpus, and for the same reason: short enough that
#: "I published a proposition" is followed by the new answer while the author is
#: still at their desk.
TTL_SECONDS = 900.0
ERROR_BACKOFF_SECONDS = 120.0

#: Propositions run 1,400 to 3,400 characters, well past what a cross-encoder
#: reads, so they are split rather than indexed whole. 900 is the corpus's own
#: chunk size, which keeps a proposition's passages comparable to a paper's.
CHUNK_CHARS = 900

#: Below this a proposition is a stub — a heading with nothing written under it
#: — and cannot support a citation.
MIN_CHARS = 120

_TIMEOUT = httpx.Timeout(8.0, connect=3.0)

_CACHE: tuple[float, str | None, list[Chunk]] | None = None
_LAST_ERROR_AT: float = 0.0

last_error: str | None = None


def _split(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Break a proposition into passages, on sentence boundaries.

    Mid-sentence splits are the thing to avoid: a passage that begins "…which
    is why the gradient vanishes" can be retrieved and cited for a claim whose
    subject was in the previous chunk.
    """
    text = text.strip()
    if len(text) <= size:
        return [text]

    # Sentence ends, keeping the punctuation with the sentence it closes.
    parts = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    buf = ""
    for p in parts:
        if buf and len(buf) + 1 + len(p) > size:
            out.append(buf)
            buf = p
        else:
            buf = f"{buf} {p}".strip()
    if buf:
        out.append(buf)
    return out


def to_chunks(payload: dict) -> list[Chunk]:
    """Adapt the published corpus into retrievable passages."""
    out: list[Chunk] = []
    for d in payload.get("documents", []):
        text = re.sub(r"\s+", " ", d.get("text", "")).strip()
        # The publisher already drops stubs, and this does not take that on
        # trust: a consumer that assumes its source filtered correctly is one
        # upstream change away from citing an empty proposition. A passage this
        # short cannot support a claim in any case.
        if len(text) < MIN_CHARS:
            continue

        ref = str(d.get("id", "")).replace("elementa:", "")
        title = re.sub(r"\s+", " ", d.get("title", "")).strip()
        book = d.get("book", "")
        chapter = d.get("chapter", "")
        url = d.get("url") or None

        for i, passage in enumerate(_split(text)):
            out.append(
                Chunk(
                    # Every passage of a proposition is separately citable, so
                    # a marker resolves to the passage that supports it rather
                    # than to the proposition as a whole.
                    chunk_id=f"elementa:{ref}:{i}",
                    doc_id=f"elementa:{ref}",
                    ordinal=i,
                    text=passage,
                    section_kind="other",
                    # Reads as "Elementa I.5.2 · Neural networks · Convolution",
                    # which is how the site cites a proposition everywhere else.
                    section_heading=" · ".join(
                        x for x in (f"Elementa {ref}", book, chapter) if x
                    ),
                    page_start=0,
                    page_end=0,
                    # Not "abstract". A proposition has no pages and is not a
                    # paper; calling it either would misdescribe what it is.
                    page_label="proposition",
                    doc_title=title,
                    url=url,
                )
            )
    return out


async def fetch(url: str | None = None, force: bool = False) -> list[Chunk]:
    """The Elementa as chunks, cached and revalidated by ETag.

    Never raises. A textbook that is briefly unreachable should cost the
    passages it would have contributed, not the answer.
    """
    global _CACHE, _LAST_ERROR_AT, last_error

    target = url or CORPUS_URL
    now = time.monotonic()

    if _CACHE is not None and not force:
        fetched_at, _etag, chunks = _CACHE
        if now - fetched_at < TTL_SECONDS:
            return chunks

    if _CACHE is None and now - _LAST_ERROR_AT < ERROR_BACKOFF_SECONDS:
        return []

    headers = {"User-Agent": "ResearchLens (+https://asifuddin.com/researchlens)"}
    if _CACHE is not None and _CACHE[1]:
        headers["If-None-Match"] = _CACHE[1]

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            r = await client.get(target, headers=headers)
        if r.status_code == 304 and _CACHE is not None:
            _CACHE = (now, _CACHE[1], _CACHE[2])
            last_error = None
            return _CACHE[2]
        r.raise_for_status()
        chunks = to_chunks(r.json())
    except Exception as e:                                   # noqa: BLE001
        _LAST_ERROR_AT = now
        last_error = f"{type(e).__name__}: {e}"
        return _CACHE[2] if _CACHE is not None else []

    last_error = None
    _CACHE = (now, r.headers.get("ETag"), chunks)
    return chunks


def reset_cache() -> None:
    """Drop the cache. For tests, and for a deliberate refresh."""
    global _CACHE, _LAST_ERROR_AT, last_error
    _CACHE = None
    _LAST_ERROR_AT = 0.0
    last_error = None
