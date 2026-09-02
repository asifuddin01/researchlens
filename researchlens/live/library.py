"""The library — papers the author adds through his CMS, indexed here.

The third fetched source, and the only one that fetches artefacts rather than
text. `/author.json` and `/elementa.json` ship prose because the site is what
holds it; a paper is a PDF, and the parser that turns one into passages already
lives in this repository and is better than anything a static site build would
do. So the site publishes a manifest of addresses and this decides what a page
of one means.

**Indexed, not learned.** Nothing here trains anything. A paper added at three
o'clock is answerable at one minute past, because retrieval reads it at question
time — and every claim drawn from it still arrives bound to a passage a reader
can open. A system that learned it instead would need retraining, and could
then paraphrase it from memory with nothing to check the paraphrase against.

**Bounded on purpose.** Each paper costs a download, a parse and an embedding
pass, paid on the first question after the manifest changes and again whenever
the process restarts — which on a free Space is often. The caps below are what
keep that cost a pause rather than an outage, and they are stated in the error
rather than enforced silently.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import replace

import httpx

from researchlens.ingest.chunk import chunk_document
from researchlens.ingest.parse import parse_bytes
from researchlens.types import Chunk, Document

MANIFEST_URL = os.getenv("LIBRARY_MANIFEST_URL", "https://asifuddin.com/library.json")

#: The same window as the Elementa, and for the same reason: short enough that
#: "I uploaded a paper" is followed by the new answer while the author is still
#: at their desk. It is affordable at that length because the expensive work is
#: conditional — an unchanged manifest answers 304 and costs one request, and
#: only a manifest that actually changed pays for downloads and parsing.
TTL_SECONDS = 900.0
ERROR_BACKOFF_SECONDS = 300.0

#: A free Space has to hold all of this in memory and rebuild it on every wake.
MAX_PAPERS = 25
MAX_BYTES = 25 * 1024 * 1024

_TIMEOUT = httpx.Timeout(30.0, connect=5.0)

#: (fetched_at, manifest_etag, documents, chunks)
_CACHE: tuple[float, str | None, list[Document], list[Chunk]] | None = None
_LAST_ERROR_AT: float = 0.0

last_error: str | None = None
#: Papers that were listed and could not be indexed, and why. Surfaced rather
#: than swallowed: a paper silently missing from the corpus is indistinguishable
#: from one the corpus has nothing to say about.
skipped: list[str] = []


async def _download(client: httpx.AsyncClient, url: str) -> bytes | None:
    try:
        r = await client.get(url)
        r.raise_for_status()
    except Exception as e:                                   # noqa: BLE001
        skipped.append(f"{url.rsplit('/', 1)[-1]}: {type(e).__name__}")
        return None
    if len(r.content) > MAX_BYTES:
        skipped.append(f"{url.rsplit('/', 1)[-1]}: larger than {MAX_BYTES // 1_048_576}MB")
        return None
    return r.content


def _index(raw: bytes, meta: dict) -> tuple[Document, list[Chunk]] | None:
    """Parse one paper into passages, or explain why it produced none."""
    name = meta.get("title") or meta["url"].rsplit("/", 1)[-1]
    try:
        doc = parse_bytes(raw, name=name, source=f"library:{meta['id']}")
    except Exception as e:                                   # noqa: BLE001
        skipped.append(f"{name}: {type(e).__name__}")
        return None

    chunks = chunk_document(doc)
    if not chunks:
        # Almost always a scan with no text layer. Saying so is more useful
        # than an empty entry the reader cannot account for.
        skipped.append(f"{name}: parsed but produced no passages (a scan?)")
        return None

    # The title the author typed into the CMS wins over the one the parser
    # inferred. The parser has to guess from typography and guesses badly on
    # anything that is not a typeset journal page: given a plain export it took
    # the entire opening paragraph as the title, which is what would then have
    # appeared in every citation and in the source browser. Here there is no
    # need to guess — somebody wrote the title down.
    title = (meta.get("title") or "").strip()
    if title and title != doc.title:
        doc = replace(doc, title=title)

    # Authors likewise, and more often than not the parser found none at all:
    # a byline is a line of ordinary text under a heading, and there is nothing
    # in a PDF that marks it as a byline. One string in the CMS, one author per
    # comma — the same shape the parser produces from a paper it does read.
    authors = [a.strip() for a in (meta.get("authors") or "").split(",") if a.strip()]
    if authors:
        doc = replace(doc, authors=authors)

    # The PDF's own address, so a citation is a link to the page it cites
    # rather than a title the reader has to go and find. The corpus cannot do
    # this — its papers are licensed journal PDFs that are not ours to host —
    # and the library can, because the author put these on his own site.
    url = meta.get("url") or None
    if url or title:
        chunks = [
            replace(
                c,
                url=url or c.url,
                doc_title=title or c.doc_title,
            )
            for c in chunks
        ]
    return doc, chunks


async def fetch(url: str | None = None, force: bool = False) -> tuple[list[Document], list[Chunk]]:
    """The library as documents and passages. Never raises."""
    global _CACHE, _LAST_ERROR_AT, last_error

    target = url or MANIFEST_URL
    now = time.monotonic()

    if _CACHE is not None and not force:
        fetched_at, _etag, docs, chunks = _CACHE
        if now - fetched_at < TTL_SECONDS:
            return docs, chunks

    if _CACHE is None and now - _LAST_ERROR_AT < ERROR_BACKOFF_SECONDS:
        return [], []

    headers = {"User-Agent": "ResearchLens (+https://asifuddin.com/researchlens)"}
    if _CACHE is not None and _CACHE[1]:
        headers["If-None-Match"] = _CACHE[1]

    skipped.clear()
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            r = await client.get(target, headers=headers)
            if r.status_code == 304 and _CACHE is not None:
                _CACHE = (now, _CACHE[1], _CACHE[2], _CACHE[3])
                last_error = None
                return _CACHE[2], _CACHE[3]
            r.raise_for_status()
            listed = r.json().get("documents", [])[:MAX_PAPERS]

            # Downloads run together; parsing does not. Parsing is CPU-bound
            # and the gather would only queue it behind itself.
            blobs = await asyncio.gather(
                *(_download(client, d["url"]) for d in listed)
            )
    except Exception as e:                                   # noqa: BLE001
        _LAST_ERROR_AT = now
        last_error = f"{type(e).__name__}: {e}"
        return (_CACHE[2], _CACHE[3]) if _CACHE is not None else ([], [])

    docs: list[Document] = []
    chunks: list[Chunk] = []
    for meta, raw in zip(listed, blobs):
        if raw is None:
            continue
        got = _index(raw, meta)
        if got is None:
            continue
        doc, cs = got
        docs.append(doc)
        chunks.extend(cs)

    last_error = None
    _CACHE = (now, r.headers.get("ETag"), docs, chunks)
    return docs, chunks


def reset_cache() -> None:
    """Drop the cache. For tests, and for a deliberate refresh."""
    global _CACHE, _LAST_ERROR_AT, last_error
    _CACHE = None
    _LAST_ERROR_AT = 0.0
    last_error = None
    skipped.clear()
