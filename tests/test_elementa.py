"""The Elementa corpus — splitting, adaptation and caching.

No network. The fetch tests drive a fake transport, so the cache, the ETag
revalidation and the failure path are exercised without depending on the site
being up; the live endpoint is checked by scripts/verify.py.
"""

import asyncio

import httpx
import pytest

from researchlens.generate.prompt import _passage_kind
from researchlens.live import elementa


PAYLOAD = {
    "author": "Md. Asif Uddin",
    "site": "https://asifuddin.com",
    "work": "Elementa",
    "schema": 1,
    "documents": [
        {
            "id": "elementa:I.1.P01",
            "kind": "proposition",
            "title": "A model is a function with a second argument.",
            "url": "https://asifuddin.com/elementa/neural-networks/computation/a-model",
            "book": "Neural networks",
            "chapter": "Computation",
            "level": "mechanics",
            "updated": "2026-08-22",
            "text": "A model is a function with a second argument. " + ("Fixing the parameters gives a particular function. " * 40),
        },
        # Too short to support a citation; must not become one.
        {"id": "elementa:I.1.P02", "title": "Stub", "text": "Too short.", "url": ""},
    ],
}


@pytest.fixture(autouse=True)
def _clean_cache():
    elementa.reset_cache()
    yield
    elementa.reset_cache()


# --- splitting --------------------------------------------------------------

def test_a_short_proposition_stays_one_passage():
    assert elementa._split("One sentence only.", 900) == ["One sentence only."]


def test_a_long_proposition_is_split():
    text = "Sentence number one. " * 200
    parts = elementa._split(text, 900)
    assert len(parts) > 1
    assert all(len(p) <= 900 for p in parts)


def test_splits_land_on_sentence_boundaries():
    """A passage beginning mid-sentence can be cited for a claim whose subject
    was in the chunk before it."""
    text = ("The gradient vanishes when the activations saturate. "
            "This is why initialisation matters. ") * 30
    for p in elementa._split(text, 400):
        assert p[0].isupper(), p[:40]
        assert p.rstrip().endswith(('.', '!', '?')), p[-40:]


def test_no_text_is_lost_in_splitting():
    text = "Alpha beta gamma. " * 120
    joined = " ".join(elementa._split(text, 500))
    assert joined.split() == text.split()


# --- adaptation -------------------------------------------------------------

def test_a_proposition_becomes_citable_passages():
    chunks = elementa.to_chunks(PAYLOAD)
    assert chunks, "the long proposition should produce passages"
    c = chunks[0]
    assert c.doc_id == "elementa:I.1.P01"
    assert c.doc_title == "A model is a function with a second argument."
    assert c.url.endswith("/a-model")


def test_every_passage_of_a_proposition_is_separately_citable():
    ids = [c.chunk_id for c in elementa.to_chunks(PAYLOAD)]
    assert len(ids) == len(set(ids)), "duplicate chunk ids cannot resolve a marker"
    assert all(i.startswith("elementa:I.1.P01:") for i in ids)


def test_the_heading_cites_it_the_way_the_site_does():
    c = elementa.to_chunks(PAYLOAD)[0]
    assert c.section_heading == "Elementa I.1.P01 · Neural networks · Computation"


def test_a_proposition_is_not_labelled_as_a_paper():
    """`pages` defaults to "abstract" for anything without page numbers, which
    would dress a textbook explanation as a scientific one."""
    c = elementa.to_chunks(PAYLOAD)[0]
    assert c.pages == "proposition"
    assert c.page_ref == "proposition"


def test_a_stub_proposition_is_dropped():
    assert all(not c.doc_id.endswith("P02") for c in elementa.to_chunks(PAYLOAD))


def test_the_model_is_told_it_is_a_textbook():
    assert _passage_kind("elementa:I.1.P01:0") == "AUTHOR'S TEXTBOOK"


def test_the_textbook_is_not_confused_with_the_website():
    """Two web-fetched sources by the same author, and they are different
    claims: one is a self-description, the other is teaching."""
    assert _passage_kind("site:profile") != _passage_kind("elementa:I.1.P01:0")


# --- fetching ---------------------------------------------------------------

def _client(handler):
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


def _ok(url):
    return httpx.Response(200, json=PAYLOAD, headers={"ETag": "e1"},
                          request=httpx.Request("GET", url))


def test_fetch_parses_and_caches(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "AsyncClient",
                        _client(lambda url, h: (calls.append(h), _ok(url))[1]))
    first = asyncio.run(elementa.fetch())
    second = asyncio.run(elementa.fetch())
    assert first and len(first) == len(second)
    assert len(calls) == 1, "a second question inside the TTL should cost nothing"


def test_not_modified_keeps_the_cached_passages(monkeypatch):
    state = {"n": 0}

    def handler(url, headers):
        state["n"] += 1
        if state["n"] == 1:
            return _ok(url)
        return httpx.Response(304, headers={"ETag": "e1"},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    asyncio.run(elementa.fetch())
    again = asyncio.run(elementa.fetch(force=True))
    assert again and elementa.last_error is None


def test_a_failed_fetch_is_reported_not_swallowed(monkeypatch):
    def handler(url, headers):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    assert asyncio.run(elementa.fetch()) == []
    assert elementa.last_error and "ConnectError" in elementa.last_error


def test_a_stale_textbook_beats_no_textbook(monkeypatch):
    state = {"n": 0}

    def handler(url, headers):
        state["n"] += 1
        if state["n"] == 1:
            return _ok(url)
        raise httpx.ConnectError("gone")

    monkeypatch.setattr(httpx, "AsyncClient", _client(handler))
    n = len(asyncio.run(elementa.fetch()))
    kept = asyncio.run(elementa.fetch(force=True))
    assert len(kept) == n
    assert elementa.last_error is not None
