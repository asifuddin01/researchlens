"""The author's library — papers added through the CMS, indexed here.

Between the two things this system already had. The corpus is a hundred and one
papers, parsed once and shipped as a prebuilt bundle; an upload is one reader's
manuscript, held for one session and thrown away. A paper the author adds to his
own site is neither: it is permanent, it is public, and it belongs in the corpus
— but the bundle is built at deploy time and the whole point is that adding a
paper should *not* require a deploy.

So it is indexed the way an upload is, in a pipeline of its own, and merged the
way an upload is. The pipeline is separate because the corpus's dense matrix and
BM25 statistics are built once over 9,593 passages, and rebuilding both on every
manifest change would cost more than every query it would improve.

**Indexed, not learned.** Adding a paper does not train anything. It becomes
retrievable — read at question time and cited to a passage a reader can open —
which is why it is answerable a minute later rather than after a training run,
and why an answer drawn from it can still be checked against it.
"""

from __future__ import annotations

import threading
import time

from researchlens.retrieval.bm25 import BM25Retriever
from researchlens.retrieval.dense import DenseRetriever
from researchlens.retrieval.pipeline import RetrievalConfig, RetrievalPipeline
from researchlens.types import Chunk, Document, Retrieved


class LibraryIndex:
    """The library, fetched and indexed, rebuilt only when it changes."""

    def __init__(
        self, embedding_model: str, reranker, corpus_doc_ids: set[str] | None = None
    ) -> None:
        self.embedding_model = embedding_model
        #: Document ids already in the bundle. A document id is the sha256 of
        #: the file's bytes, so uploading a paper the corpus already has is
        #: detectable rather than a matter of comparing titles — and it is
        #: dropped, because two copies of one paper would let a single source
        #: fill two evidence slots and read as corroboration.
        self.corpus_doc_ids = corpus_doc_ids or set()
        #: The corpus reranker, borrowed. A cross-encoder scores a (query,
        #: passage) pair on its own terms and holds no per-corpus state, so
        #: sharing it is what keeps a library passage's score comparable with
        #: a corpus passage's — the same reason uploads borrow it.
        self.reranker = reranker

        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.pipeline: RetrievalPipeline | None = None
        self.last_error: str | None = None
        self.skipped: list[str] = []
        self.indexed_at: float | None = None

        #: What the current index was built from. A manifest that fetches
        #: unchanged must not cost an embedding pass.
        self._fingerprint: tuple[str, ...] = ()
        #: A refresh mutates the pipeline a query may be reading, and both
        #: arrive on the server's thread pool.
        self._lock = threading.Lock()

    # ---- state -----------------------------------------------------------

    @property
    def doc_ids(self) -> set[str]:
        return {d.doc_id for d in self.documents}

    def stats(self) -> dict:
        """What /health reports: enough to tell empty from broken."""
        return {
            "documents": len(self.documents),
            "passages": len(self.chunks),
            "skipped": list(self.skipped),
            "error": self.last_error,
            "indexed_at": self.indexed_at,
        }

    # ---- refresh ---------------------------------------------------------

    async def refresh(self, force: bool = False) -> None:
        """Fetch the manifest and rebuild if the library changed.

        Never raises. A library that is briefly unreachable costs the passages
        it would have contributed, not the answer — and the previous index is
        kept rather than cleared, because a stale paper still answers better
        than no paper.
        """
        from researchlens.live import library as source

        docs, chunks = await source.fetch(force=force)
        self.last_error = source.last_error
        self.skipped = list(source.skipped)

        docs, chunks = self._dedupe(docs, chunks)

        fingerprint = tuple(c.chunk_id for c in chunks)
        if fingerprint == self._fingerprint and self.pipeline is not None:
            return
        if not chunks:
            # Distinguish "the author has added nothing" from "the fetch
            # failed": the first empties the index, the second keeps it.
            if self.last_error is None:
                with self._lock:
                    self.documents, self.chunks = [], []
                    self.pipeline, self._fingerprint = None, ()
            return

        pipeline = RetrievalPipeline(
            # Caching is on, unlike a session upload's. These papers are the
            # author's own and public, and the cache key is derived from the
            # passages themselves — so a changed library gets a new key on its
            # own, and an unchanged one costs no embedding pass after a
            # restart. On a Space that sleeps, that is most restarts.
            dense=DenseRetriever(model=self.embedding_model),
            bm25=BM25Retriever(),
            reranker=self.reranker,
        )
        pipeline.index(chunks)

        with self._lock:
            self.documents, self.chunks = docs, chunks
            self.pipeline = pipeline
            self._fingerprint = fingerprint
            self.indexed_at = time.time()

    def _dedupe(
        self, docs: list[Document], chunks: list[Chunk]
    ) -> tuple[list[Document], list[Chunk]]:
        """Drop what the corpus already has, and any paper listed twice.

        The second case is not hypothetical: the manifest lists whatever is in
        the CMS, and the same PDF uploaded under two entries — a preprint and
        its published version, say — parses to the same document id and the
        same chunk ids. `RetrievalPipeline.index` refuses duplicate ids outright
        because row indices would stop matching the retrievers, so an
        unfiltered manifest would take down the whole library index rather than
        just the duplicate entry.
        """
        keep: list[Document] = []
        seen: set[str] = set()
        for d in docs:
            if d.doc_id in self.corpus_doc_ids:
                self.skipped.append(f"{d.title or d.doc_id}: already in the indexed corpus")
                continue
            if d.doc_id in seen:
                self.skipped.append(f"{d.title or d.doc_id}: listed more than once")
                continue
            seen.add(d.doc_id)
            keep.append(d)

        if len(keep) == len(docs):
            return docs, chunks

        # By chunk id, not by document id. Two entries for one PDF parse to the
        # same document *and* the same chunk ids, so a document-id filter keeps
        # both copies of every passage — which is the exact duplication this
        # exists to prevent, and it raised on the first test that listed one
        # paper twice.
        kept: list[Chunk] = []
        seen_chunks: set[str] = set()
        for c in chunks:
            if c.doc_id not in seen or c.chunk_id in seen_chunks:
                continue
            seen_chunks.add(c.chunk_id)
            kept.append(c)
        return keep, kept

    # ---- search ----------------------------------------------------------

    def search(
        self, query: str, config: RetrievalConfig, doc_ids: set[str] | None = None
    ) -> list[Retrieved]:
        """Retrieve from the library. Empty when there is nothing indexed."""
        with self._lock:
            pipeline = self.pipeline
            mine = self.doc_ids
        if pipeline is None:
            return []
        if doc_ids is not None and not (doc_ids & mine):
            # The reader chose papers and none of them is in the library.
            return []
        return [
            Retrieved(
                chunk=r.chunk, score=r.score, dense_score=r.dense_score,
                bm25_score=r.bm25_score, rrf_score=r.rrf_score,
                rerank_score=r.rerank_score,
                sources=r.sources | {"library"},
            )
            for r in pipeline.search(query, config, doc_ids)
        ]
