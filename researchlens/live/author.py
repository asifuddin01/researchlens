"""The author corpus — what the site says about the person who built this.

Every other source here answers questions about a literature. This one answers
the question a visitor actually arrives with: who wrote this, and what do they
work on. Without it the system did the most annoying thing it can do — refuse,
correctly, because it had no evidence, to a question whose answer was sitting
on the same domain the page was served from.

**It is fetched, not bundled.** The site publishes `/author.json`, rebuilt by
`astro build` on every push to main — Sveltia's /admin commits included. So a
paper added through the CMS, or a project blurb edited, changes what this
system knows within one cache window and without a redeploy on either side.
Bundling the same text into this repository would make it a copy that goes
stale silently, which is the failure mode the whole design is arranged to
avoid: an answer that looks exactly as confident when its evidence is a year
out of date.

**It is evidence like any other.** The documents become `Chunk`s, compete
through the same cross-encoder, and are cited with a URL a reader can open. No
part of the pipeline gets a special case for them beyond the two that concern
honesty: the passage label says the text came from a personal website, and the
citation says "asifuddin.com" rather than borrowing the vocabulary of a paper.
That matters because a self-description is not a peer-reviewed finding, and the
reader is entitled to see which one they are being shown.
"""

from __future__ import annotations

import os
import re
import time

import httpx

from researchlens.types import Chunk

#: Overridable so a local build can point at a dev server, and so this file
#: does not hard-code somebody's domain into the retrieval path forever.
CORPUS_URL = os.getenv("AUTHOR_CORPUS_URL", "https://asifuddin.com/author.json")

#: How long a fetched corpus is trusted. Fifteen minutes is short enough that
#: "I updated my site" is followed by the new answer while the person is still
#: at their desk, and long enough that a burst of questions is one request.
TTL_SECONDS = 900.0

#: A failed fetch is not retried on every question. Without this, a site that
#: is down turns each author question into a fresh timeout, and the visible
#: symptom is a slow refusal rather than a fast one.
ERROR_BACKOFF_SECONDS = 120.0

_TIMEOUT = httpx.Timeout(6.0, connect=3.0)

#: (fetched_at, etag, chunks). Module level so every Engine in the process
#: shares one cache; the corpus is the same for all of them.
_CACHE: tuple[float, str | None, list[Chunk]] | None = None
_LAST_ERROR_AT: float = 0.0

#: The author's name, taken from the corpus rather than hard-coded, so this
#: file works for whoever publishes an /author.json. Only used to resolve
#: anaphora, and only after a successful fetch has supplied it.
_NAME: str = ""


def name() -> str:
    """The author's name, once a corpus has been fetched. Empty before."""
    return _NAME

#: Set when a fetch fails, so the caller can say why rather than returning
#: nothing and letting silence look like "the site had nothing to say".
last_error: str | None = None


# --------------------------------------------------------------------------
# Which questions this source is for
# --------------------------------------------------------------------------

#: Words that make a question about the author rather than about a literature.
#:
#: Deliberately generous. A false positive costs a handful of chunks that the
#: reranker's relevance floor then drops, because a passage about somebody's
#: education does not score well against a question on retrieval augmentation.
#: A false negative costs a refusal to a question the system could have
#: answered, which is far worse and much harder for a reader to diagnose.
_AUTHOR_WORDS = (
    "asif", "the author", "author's", "authors'",
    "who built", "who made", "who wrote this", "who created",
    "your background", "your research", "your work", "your experience",
    "his background", "his research", "his work", "his experience",
    "their background", "who is behind", "about you", "about the person",
    "portfolio", "asifuddin",
)

#: Phrases that ask after a person's standing rather than a fact in a paper —
#: strengths, experience, availability. On their own these are ambiguous, so
#: they only count alongside a pronoun or name that points at a person.
_PERSONAL_TOPICS = (
    "strength", "strengths", "weakness", "skills", "expertise", "cv",
    "resume", "curriculum vitae", "education", "degree", "supervisor",
    "hire", "hiring", "available", "contact", "email", "affiliation",
    "publications", "background", "experience", "currently working",
    "working on", "recently", "bio", "biography",
)

_PERSON_HINTS = ("he ", "his ", "him ", "they ", "their ", "you ", "your ", "asif", "author")


def asks_about_the_author(question: str) -> bool:
    """Whether the site is a source worth consulting for this question.

    Two ways to qualify. Either the question names the author outright, or it
    asks after something that is only ever true of a person — strengths,
    a degree, who supervised the work — while pointing at one.
    """
    q = f" {question.lower().strip()} "
    if any(w in q for w in _AUTHOR_WORDS):
        return True
    if any(t in q for t in _PERSONAL_TOPICS) and any(h in q for h in _PERSON_HINTS):
        return True
    return False


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


_ANAPHORA = (
    # "the author's", "authors'", "author" — the phrase a visitor reaches for
    # when they are looking at a page and do not know whose it is.
    (re.compile(r"\b(?:the\s+)?authors?['\u2019]?s?\b", re.I), "{name}"),
    # Possessives before plain pronouns, or "his" becomes "NAME s".
    (re.compile(r"\b(?:his|her|their|your)\b", re.I), "{name}'s"),
    (re.compile(r"\b(?:he|she|they|you)\b", re.I), "{name}"),
)


def resolve_anaphora(question: str, who: str | None = None) -> str:
    """Put the author's name where the question used a pronoun.

    This is the difference between the source working and not working, and it
    is worth stating why rather than leaving it as a tidy-up.

    The reranker is a cross-encoder trained on search relevance. It has no way
    to know that "the author" denotes the person a document is about, so it
    scores a biography against "what are the author's strengths" on surface
    overlap and finds almost none — measured at -9.17, below a book review
    about causal inference, and below the *corpus*, whose best passage for that
    question was itself irrelevant at -4.77. The site was being dropped for
    losing a comparison it never entered.

    Naming the person repairs it. The same question, with "the author" replaced
    by the name, ranks the profile at +3.94; "what research does the author do"
    goes from -5.02 to +8.75, with the research plates directly behind it. A
    swing of thirteen points from resolving one anaphor.

    Applied only to the site pool, and only on questions already judged to be
    about the author. The corpus is ranked against what the reader actually
    typed, because there a pronoun means whatever the passage means by it.
    """
    who = who or _NAME
    if not who:
        return question
    out = question
    for pattern, template in _ANAPHORA:
        out = pattern.sub(template.format(name=who), out)
    # Nothing matched and no part of the name is present: say it anyway, so
    # the pool is ranked against a query that names its subject.
    #
    # Any token, not the last one: "recently what is Asif doing?" already names
    # him by his first name, and prefixing the full name there would rank the
    # pool against a question nobody asked. Honorifics and initials are too
    # short to be evidence of anything, so they do not count.
    parts = [t.strip(".,'\u2019").lower() for t in who.split()]
    if not any(len(t) > 2 and t in out.lower() for t in parts):
        out = f"{who}: {out}"
    return re.sub(r"\s+", " ", out).strip()


def to_chunks(payload: dict) -> list[Chunk]:
    """Adapt the site's JSON into the evidence type the pipeline speaks.

    One document becomes one chunk. They are already short and already
    self-contained — the endpoint writes each as a paragraph that reads on its
    own — so splitting them further would only sever a sentence from the name
    it belongs to.
    """
    global _NAME
    author = _clean(payload.get("author", "")) or "the author"
    _NAME = author
    out: list[Chunk] = []
    for i, d in enumerate(payload.get("documents", [])):
        text = _clean(d.get("text", ""))
        if not text:
            continue
        doc_id = f"site:{d.get('id', i)}"
        kind = _clean(d.get("kind", "page"))
        updated = _clean(d.get("updated", ""))
        out.append(
            Chunk(
                chunk_id=doc_id,
                doc_id=doc_id,
                ordinal=i,
                text=text,
                # Not "abstract": this is not a paper and must not read as one.
                section_kind="other",
                # The heading is what a citation shows, so it names the site,
                # the sort of page, and — where the site tracks one — when it
                # last changed. A reader can then judge for themselves whether
                # a self-description is the evidence they wanted.
                section_heading=_clean(
                    f"asifuddin.com · {kind}" + (f" · updated {updated}" if updated else "")
                ),
                page_start=0,
                page_end=0,
                page_label="website",
                doc_title=_clean(d.get("title", "")) or author,
                url=_clean(d.get("url", "")) or None,
            )
        )
    return out


async def fetch(url: str | None = None, force: bool = False) -> list[Chunk]:
    """The author corpus as chunks, cached and conditionally revalidated.

    Returns [] and sets `last_error` on failure. Never raises: this is an
    enhancement to what the system can answer, and a site that is briefly
    unreachable should cost a question about the author, not every question.
    """
    global _CACHE, _LAST_ERROR_AT, last_error

    target = url or CORPUS_URL
    now = time.monotonic()

    if _CACHE is not None and not force:
        fetched_at, _etag, chunks = _CACHE
        if now - fetched_at < TTL_SECONDS:
            return chunks

    # A recent failure with nothing cached: do not retry yet. With a stale
    # cache we would rather serve slightly old facts than none, so the backoff
    # only applies when there is nothing to fall back to.
    if _CACHE is None and now - _LAST_ERROR_AT < ERROR_BACKOFF_SECONDS:
        return []

    headers = {"User-Agent": "ResearchLens (+https://asifuddin.com/researchlens)"}
    if _CACHE is not None and _CACHE[1]:
        headers["If-None-Match"] = _CACHE[1]

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            r = await client.get(target, headers=headers)
        # Unchanged since last time: keep the chunks, restart the clock. This
        # is the common case once the site settles, and it costs no parsing.
        if r.status_code == 304 and _CACHE is not None:
            _CACHE = (now, _CACHE[1], _CACHE[2])
            last_error = None
            return _CACHE[2]
        r.raise_for_status()
        chunks = to_chunks(r.json())
    except Exception as e:                                   # noqa: BLE001
        _LAST_ERROR_AT = now
        last_error = f"{type(e).__name__}: {e}"
        # A stale corpus beats no corpus. Say the fetch failed, keep answering.
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
