"""Chunking tests.

The invariant under test throughout: a chunk never spans two sections, and it
always knows where it came from. Everything about citation correctness rests on
that, so it is checked directly rather than inferred from an end-to-end result.
"""

from researchlens.ingest.chunk import chunk_document, fingerprint
from researchlens.types import Document, Section


def _doc(*sections: Section) -> Document:
    return Document(
        doc_id="deadbeef",
        title="A Paper",
        authors=["Author One"],
        sections=list(sections),
        n_pages=9,
        source_path="/tmp/a.pdf",
    )


def _section(kind, heading, p0, p1, text):
    return Section(kind=kind, heading=heading, page_start=p0, page_end=p1, text=text)


SENT = "This sentence carries a claim about kidney segmentation performance. "


def test_chunks_never_span_sections():
    doc = _doc(
        _section("methods", "2 Methods", 3, 4, SENT * 12),
        _section("results", "3 Results", 5, 7, SENT * 12),
    )
    chunks = chunk_document(doc, size=400, overlap=80)
    assert len({c.section_kind for c in chunks}) == 2
    for c in chunks:
        # A chunk's pages must lie inside its own section's span.
        if c.section_kind == "methods":
            assert (c.page_start, c.page_end) == (3, 4)
        else:
            assert (c.page_start, c.page_end) == (5, 7)


def test_every_chunk_can_cite_itself():
    doc = _doc(_section("results", "3 Results", 7, 8, SENT * 10))
    for c in chunk_document(doc, size=400):
        assert c.doc_title == "A Paper"
        assert c.section_heading == "3 Results"
        assert c.pages == "7-8"
        assert c.chunk_id.startswith("deadbeef:")


def test_references_are_skipped_by_default():
    """A bibliography matches almost any query lexically and can never support
    a claim about what a paper found."""
    doc = _doc(
        _section("results", "Results", 6, 6, SENT * 6),
        _section("references", "References", 9, 9, "Smith J. et al. Nature 2019. " * 30),
    )
    kinds = {c.section_kind for c in chunk_document(doc)}
    assert "references" not in kinds
    assert "results" in kinds


def test_ordinals_are_contiguous_and_ids_unique():
    doc = _doc(
        _section("methods", "Methods", 2, 3, SENT * 10),
        _section("results", "Results", 4, 5, SENT * 10),
    )
    chunks = chunk_document(doc, size=400)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_overlap_carries_text_between_neighbours():
    doc = _doc(_section("results", "Results", 1, 1, SENT * 20))
    chunks = chunk_document(doc, size=300, overlap=120)
    assert len(chunks) > 1
    # Consecutive chunks in the same section should share their boundary text.
    first_tail = chunks[0].text[-60:]
    assert first_tail in chunks[1].text


def test_fragments_too_short_to_carry_a_claim_are_dropped():
    doc = _doc(_section("other", "Note", 1, 1, "Too short."))
    assert chunk_document(doc) == []


def test_fingerprint_tracks_the_parameters_labels_were_written_under():
    assert fingerprint(1000, 180) == "c1000-o180"
    assert fingerprint(800, 100) != fingerprint(1000, 180)
