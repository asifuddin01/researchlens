"""The author corpus — routing, anaphora, adaptation and caching.

No network. The fetch tests drive a fake transport so the cache, the ETag
revalidation and the failure path are all exercised without depending on a
site being up; the real endpoint is checked by scripts/verify.py.
"""

import asyncio

import httpx
import pytest

from researchlens.generate.prompt import _passage_kind
from researchlens.live import author


PAYLOAD = {
    "author": "Md. Asif Uddin",
    "site": "https://asifuddin.com",
    "schema": 1,
    "documents": [
        {
            "id": "profile", "kind": "profile", "url": "https://asifuddin.com",
            "title": "Md. Asif Uddin — who he is",
            "text": "Md. Asif Uddin is a deep learning researcher in Dhaka.",
        },
        {
            "id": "note", "kind": "review", "url": "https://asifuddin.com/marginalia/x",
            "title": "A review", "updated": "2026-08-22",
            "text": "A note about somebody else's paper.",
        },
        # No text: must be dropped rather than indexed as an empty passage.
        {"id": "empty", "kind": "page", "title": "Nothing", "text": "   "},
    ],
}


@pytest.fixture(autouse=True)
def _clean_cache():
    author.reset_cache()
    yield
    author.reset_cache()


# --- which questions reach this source --------------------------------------

@pytest.mark.parametrize("q", [
    "who is Asif?",
    "recently what is Asif doing?",
    "what research does the author do?",
    "what are the author's strengths?",
    "what is his educational background?",
    "who built this?",
    "tell me about the person behind this site",
    "what are your strengths",
])
def test_author_questions_are_routed_to_the_site(q):
    assert author.asks_about_the_author(q)


@pytest.mark.parametrize("q", [
    "how does retrieval-augmented generation reduce hallucination?",
    "what approaches exist for privacy-preserving diabetic retinopathy classification?",
    "do deep-learning models outperform linear baselines at predicting perturbation effects?",
    "how do vision-language models align image and text representations?",
])
def test_technical_questions_do_not_reach_the_site(q):
    """A false positive here puts biography in a scientific answer, which is
    the one way this feature could make every other answer worse."""
    assert not author.asks_about_the_author(q)


# --- anaphora ---------------------------------------------------------------

def test_pronouns_are_replaced_with_the_name():
    out = author.resolve_anaphora("what are his strengths?", "Md. Asif Uddin")
    assert "Md. Asif Uddin's" in out
    assert " his " not in f" {out} "


def test_the_author_becomes_the_name():
    out = author.resolve_anaphora("what research does the author do?", "Md. Asif Uddin")
    assert out == "what research does Md. Asif Uddin do?"


def test_a_question_already_naming_him_is_left_alone():
    q = "recently what is Asif doing?"
    assert author.resolve_anaphora(q, "Md. Asif Uddin") == q


def test_a_question_naming_nobody_gets_the_name_prefixed():
    """Without this the pool is ranked against a query with no subject, which
    is how "what are the strengths" scored a biography below a book review."""
    out = author.resolve_anaphora("what are the strengths", "Md. Asif Uddin")
    assert out.startswith("Md. Asif Uddin:")


def test_no_name_yet_is_not_an_error():
    """Before the first successful fetch there is no name to substitute."""
    assert author.resolve_anaphora("who is he?", "") == "who is he?"


# --- adaptation -------------------------------------------------------------

def test_documents_become_citable_chunks():
    chunks = author.to_chunks(PAYLOAD)
    assert [c.chunk_id for c in chunks] == ["site:profile", "site:note"]
    first = chunks[0]
    assert first.doc_title == "Md. Asif Uddin — who he is"
    assert first.url == "https://asifuddin.com"
    assert first.section_heading == "asifuddin.com · profile"


def test_a_document_with_no_text_is_dropped():
    """An empty passage cannot support a citation, so it must never become one."""
    assert all(c.chunk_id != "site:empty" for c in author.to_chunks(PAYLOAD))


def test_the_update_date_travels_into_the_citation():
    note = next(c for c in author.to_chunks(PAYLOAD) if c.chunk_id == "site:note")
    assert note.section_heading == "asifuddin.com · review · updated 2026-08-22"


def test_a_website_is_not_labelled_as_an_abstract():
    """`pages` defaults to "abstract" for anything without page numbers, which
    would dress a self-description as a scientific one."""
    c = author.to_chunks(PAYLOAD)[0]
    assert c.pages == "website"
    assert c.page_ref == "website"


def test_the_model_is_told_the_site_is_not_a_paper():
    assert _passage_kind("site:profile") == "AUTHOR'S OWN WEBSITE"


@pytest.mark.parametrize("cid", ["arxiv:1", "pubmed:2", "openalex:W3"])
def test_every_live_source_is_labelled_abstract_only(cid):
    """Regression: this test used to pass only for arXiv. PubMed and OpenAlex
    abstracts reached the model labelled "passage" — the exact confusion the
    label exists to prevent, in two of the three live sources."""
    assert _passage_kind(cid) == "ABSTRACT ONLY"


def test_the_name_is_taken_from_the_payload_not_hardcoded():
    author.to_chunks({"author": "Ada Lovelace", "documents": []})
    assert author.name() == "Ada Lovelace"


# --- fetching ---------------------------------------------------------------

def _client(handler):
    """Patch httpx.AsyncClient so fetch() talks to `handler` instead of a host."""
    class _Fake:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            return handler(url, headers or {})

    return _Fake


def test_fetch_parses_and_caches(monkeypatch):
    calls = []

    def handler(url, headers):
        calls.append(headers)
        return httpx.Response(200, json=PAYLOAD, headers={"ETag": "abc"},
                                  request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    first = asyncio.run(author.fetch())
    second = asyncio.run(author.fetch())
    assert len(first) == 2 and len(second) == 2
    # One request, not two: the second question inside the TTL costs nothing.
    assert len(calls) == 1


def test_a_forced_refresh_sends_the_etag(monkeypatch):
    seen = []

    def handler(url, headers):
        seen.append(headers.get("If-None-Match"))
        return httpx.Response(200, json=PAYLOAD, headers={"ETag": "abc"},
                                  request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    asyncio.run(author.fetch())
    asyncio.run(author.fetch(force=True))
    assert seen == [None, "abc"]


def test_not_modified_keeps_the_cached_chunks(monkeypatch):
    state = {"n": 0}

    def handler(url, headers):
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(200, json=PAYLOAD, headers={"ETag": "abc"},
                                  request=httpx.Request("GET", url))
        return httpx.Response(304, headers={"ETag": "abc"},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    asyncio.run(author.fetch())
    again = asyncio.run(author.fetch(force=True))
    assert len(again) == 2
    assert author.last_error is None


def test_a_failed_fetch_is_reported_not_swallowed(monkeypatch):
    def handler(url, headers):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    assert asyncio.run(author.fetch()) == []
    # Silence would be indistinguishable from "the site had nothing to say".
    assert author.last_error and "ConnectError" in author.last_error


def test_a_stale_corpus_beats_no_corpus(monkeypatch):
    """A site that goes down after a successful fetch should cost freshness,
    not the ability to answer at all."""
    state = {"n": 0}

    def handler(url, headers):
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(200, json=PAYLOAD, headers={"ETag": "abc"},
                                  request=httpx.Request("GET", url))
        raise httpx.ConnectError("gone")

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    asyncio.run(author.fetch())
    kept = asyncio.run(author.fetch(force=True))
    assert len(kept) == 2
    assert author.last_error is not None


def test_repeated_failures_back_off(monkeypatch):
    """Without a backoff, a down site turns every author question into a fresh
    timeout — a slow refusal instead of a fast one."""
    calls = []

    def handler(url, headers):
        calls.append(url)
        raise httpx.ConnectError("gone")

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    asyncio.run(author.fetch())
    asyncio.run(author.fetch())
    assert len(calls) == 1
