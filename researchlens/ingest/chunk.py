"""Chunking that carries structure through to retrieval.

Chunks are cut *within* sections, never across them. A passage spanning the end
of Methods and the start of Results cannot honestly cite either, and mixed
passages are the usual reason a retrieved chunk looks relevant to a reader and
scores badly against a hand-written label.

Sizes are in characters rather than tokens on purpose. Token counts differ
between the embedding model, the reranker and whichever generator is answering,
so a token-exact boundary is exact for one component and approximate for the
other three. Characters are wrong for all of them equally, cost no tokeniser
dependency at ingest, and stay stable when a model is swapped — which the
provider abstraction exists to make routine.
"""

from __future__ import annotations

import re

from researchlens.types import Chunk, Document

#: Roughly 200-250 tokens of English prose. Large enough to hold a claim with
#: its evidence, small enough that a reranker sees mostly signal.
DEFAULT_SIZE = 1000

#: One or two sentences of carry-over, so a claim split across a boundary is
#: recoverable from either side.
DEFAULT_OVERLAP = 180

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, conservatively.

    The lookahead requires a capital, an opening bracket or a paren after the
    space, which keeps "et al. (2021)" and "Fig. 3" and "0.94 vs. 0.91" intact.
    Abbreviations still break occasionally; the overlap covers the damage, and
    a real sentence segmenter is a dependency this does not need.
    """
    parts = [p.strip() for p in _SENTENCE_END.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_document(
    doc: Document,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    skip_sections: frozenset[str] = frozenset({"references"}),
) -> list[Chunk]:
    """Cut one document into retrievable passages.

    References are skipped by default. A bibliography is a dense field of
    author names, venues and years that matches almost any query lexically —
    it is the single biggest source of plausible-looking BM25 false positives,
    and it can never support a claim about what a paper *found*.
    """
    chunks: list[Chunk] = []
    ordinal = 0

    for section in doc.sections:
        if section.kind in skip_sections:
            continue

        sentences = _split_sentences(section.text)
        buf: list[str] = []
        buf_len = 0

        def flush() -> None:
            nonlocal ordinal, buf, buf_len
            text = " ".join(buf).strip()
            if len(text) < 80:  # too short to carry a claim
                buf, buf_len = [], 0
                return
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{ordinal}",
                    doc_id=doc.doc_id,
                    ordinal=ordinal,
                    text=text,
                    section_kind=section.kind,
                    section_heading=section.heading,
                    # Page attribution is at section granularity. Going finer
                    # means tracking which page each sentence landed on through
                    # the line grouping, which is real work for a citation that
                    # already says "Results, pp. 7-8" — precise enough for a
                    # reader to find the passage on the page.
                    page_start=section.page_start,
                    page_end=section.page_end,
                    doc_title=doc.title,
                )
            )
            ordinal += 1

            # Carry the tail sentences forward as overlap.
            tail: list[str] = []
            tail_len = 0
            for s in reversed(buf):
                if tail_len + len(s) > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s) + 1
            buf, buf_len = tail, tail_len

        for sentence in sentences:
            # A single sentence longer than the target is not split further.
            # Long sentences in papers are usually enumerations or equations,
            # and cutting mid-clause produces two passages that each read as
            # gibberish to a reranker.
            if buf and buf_len + len(sentence) > size:
                flush()
            buf.append(sentence)
            buf_len += len(sentence) + 1

        flush()

    return chunks


def chunk_corpus(docs: list[Document], **kw) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, **kw))
    return out


def fingerprint(size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP) -> str:
    """Identify the chunking parameters that produced an index.

    Hand-written ground truth in `eval/questions.jsonl` names chunk ids, and
    those ids only mean anything under the parameters that generated them.
    The harness compares this fingerprint against the one recorded with the
    labels and refuses to score if they differ, rather than reporting a
    confidently wrong Recall@5.
    """
    return f"c{size}-o{overlap}"
