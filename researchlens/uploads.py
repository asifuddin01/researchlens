"""Papers a reader adds at run time.

Kept deliberately separate from the corpus, for a reason that is easy to miss
until it has already happened: the Space is *one Python process serving every
visitor*. Appending an uploaded paper to the shared index would put it in
everybody's results — a stranger's unpublished manuscript quoted back at
someone who never uploaded it, with a citation, indistinguishable from a paper
that belongs there. That is a disclosure bug wearing the costume of a feature.

So an upload lives in a per-session index of its own and is merged into the
evidence at query time, the same way live search results are. Nothing is
written to disk, nothing outlives the session, and nothing crosses between
sessions.

The cost of that choice is that BM25 statistics over three papers are nearly
meaningless — idf cannot discriminate across a collection of forty passages.
It is still worth running: the lexical half exists to find exact strings, and
"SwinUNETR" appears in an uploaded paper or it does not. The cross-encoder does
the ranking that matters, and it is the same one the corpus goes through, so
scores from the two indexes are on one scale.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from researchlens.ingest.chunk import chunk_document
from researchlens.ingest.parse import parse_bytes
from researchlens.retrieval.bm25 import BM25Retriever
from researchlens.retrieval.dense import DenseRetriever
from researchlens.retrieval.pipeline import RetrievalConfig, RetrievalPipeline
from researchlens.types import Chunk, Document, Retrieved

#: A journal PDF with figures runs 5-15 MB. Twenty is generous for one paper
#: and small enough that a mistaken upload of something else fails fast rather
#: than spending a minute of the process's memory finding out.
MAX_BYTES = 20 * 1024 * 1024

#: Parsing is seconds per paper and every passage is embedded. Five is enough
#: to ask a real question across a reading list; a hundred is someone using a
#: shared demo as a private index.
MAX_PAPERS_PER_SESSION = 5

#: A page count guard, separate from the byte guard. A 400-page thesis is well
#: under 20 MB and would take a minute to parse while the process serves
#: nobody else.
MAX_PAGES = 80

#: How long an idle session's papers are kept. Long enough to read an answer,
#: think, and ask again; short enough that a busy afternoon does not accumulate
#: a hundred abandoned indexes in a container with no disk.
TTL_SECONDS = 60 * 60

#: Sessions retained at once, oldest evicted first. A second bound because TTL
#: alone does not help if fifty people arrive in the same minute.
MAX_SESSIONS = 40


class UploadError(ValueError):
    """A refused upload, with a reason meant for the person who uploaded it."""


@dataclass
class Session:
    """One reader's papers, and the index over them."""

    pipeline: RetrievalPipeline
    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    touched: float = field(default_factory=time.monotonic)

    @property
    def doc_ids(self) -> set[str]:
        return {d.doc_id for d in self.documents}


class UploadStore:
    """Per-session uploaded papers. Nothing here is shared or persisted."""

    def __init__(
        self,
        embedding_model: str,
        reranker,
        corpus_doc_ids: set[str] | None = None,
        max_papers: int = MAX_PAPERS_PER_SESSION,
        ttl: float = TTL_SECONDS,
        max_sessions: int = MAX_SESSIONS,
    ) -> None:
        self.embedding_model = embedding_model
        #: The corpus reranker, borrowed rather than rebuilt. It holds no
        #: per-corpus state — a cross-encoder scores a (query, passage) pair on
        #: its own terms — so one instance can serve every session, and its
        #: scores stay comparable with the corpus's.
        self.reranker = reranker
        self.corpus_doc_ids = corpus_doc_ids or set()
        self.max_papers = max_papers
        self.ttl = ttl
        self.max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}
        # Uploads mutate a session while a query may be reading it, and both
        # arrive on the server's thread pool.
        self._lock = threading.Lock()

    # ---- lifecycle -------------------------------------------------------

    def _evict(self) -> None:
        now = time.monotonic()
        for sid in [s for s, v in self._sessions.items() if now - v.touched > self.ttl]:
            del self._sessions[sid]
        while len(self._sessions) > self.max_sessions:
            oldest = min(self._sessions, key=lambda s: self._sessions[s].touched)
            del self._sessions[oldest]

    def _session(self, session_id: str, create: bool = False) -> Session | None:
        s = self._sessions.get(session_id)
        if s is None and create:
            s = Session(
                pipeline=RetrievalPipeline(
                    # No on-disk vector cache: these vectors belong to one
                    # reader for one session, and writing them beside the
                    # corpus would leave a stranger's paper on the volume.
                    dense=DenseRetriever(model=self.embedding_model, cache=False),
                    bm25=BM25Retriever(),
                    reranker=self.reranker,
                )
            )
            self._sessions[session_id] = s
        if s is not None:
            s.touched = time.monotonic()
        return s

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    # ---- adding ----------------------------------------------------------

    def add(self, session_id: str, raw: bytes, filename: str) -> Document:
        """Parse, chunk and index one uploaded PDF. Returns its `Document`.

        Raises `UploadError` with a reason a reader can act on. A scanned paper
        is the common one and the message says so — the parser already
        distinguishes "no text layer" from "no sections", and passing that
        distinction through is the difference between "OCR it first" and a
        shrug.
        """
        if not raw:
            raise UploadError("that file is empty")
        if len(raw) > MAX_BYTES:
            raise UploadError(
                f"{filename} is {len(raw) / 1e6:.0f} MB; the limit is "
                f"{MAX_BYTES // 1024 // 1024} MB"
            )
        if raw[:5] != b"%PDF-":
            # Checked by content, not by extension: the parser's error for a
            # renamed .docx is a pdfplumber traceback, which tells the reader
            # nothing.
            raise UploadError(f"{filename} is not a PDF")

        with self._lock:
            # Evicted *after* this session is registered, not before: running
            # the cap first counts a set that does not yet include the arrival,
            # so the limit is off by one and never actually binds. The session
            # created here cannot evict itself — it is the most recently
            # touched, and eviction takes the oldest.
            session = self._session(session_id, create=True)
            self._evict()
            assert session is not None
            if len(session.documents) >= self.max_papers:
                raise UploadError(
                    f"you have {len(session.documents)} papers open; "
                    f"the limit is {self.max_papers} at a time"
                )

        try:
            doc = parse_bytes(raw, name=filename, source=f"upload:{filename}")
        except ValueError as e:
            raise UploadError(str(e)) from e
        except Exception as e:  # noqa: BLE001 — a malformed PDF is a user error
            raise UploadError(f"{filename} could not be read: {e}") from e

        if doc.n_pages > MAX_PAGES:
            raise UploadError(
                f"{filename} has {doc.n_pages} pages; the limit is {MAX_PAGES}. "
                "This is a demo on shared hardware, not a document store."
            )
        if doc.doc_id in self.corpus_doc_ids:
            raise UploadError(
                f"{doc.title or filename} is already in the indexed corpus — "
                "ask about it directly."
            )

        chunks = chunk_document(doc)
        if not chunks:
            raise UploadError(
                f"{filename} parsed but produced no passages; it may be a "
                "cover sheet or a poster rather than a paper."
            )

        with self._lock:
            session = self._session(session_id, create=True)
            assert session is not None
            if doc.doc_id in session.doc_ids:
                # Same bytes, so the same document id and the same chunk ids.
                # Re-indexing would raise on duplicate ids; saying so is
                # friendlier and true.
                raise UploadError(f"{doc.title or filename} is already open")
            session.documents.append(doc)
            session.chunks.extend(chunks)
            # Rebuilt over all of the session's chunks rather than appended to.
            # BM25's idf and average length are corpus statistics: adding a
            # second paper changes the score of every passage in the first, and
            # an incremental path that pretended otherwise would rank against
            # numbers that no longer hold. At a few hundred passages a rebuild
            # is milliseconds plus one embedding pass over the new paper only —
            # which the dense cache inside the process does not give us, so it
            # is the one real cost, and it is bounded by MAX_PAPERS.
            session.pipeline.index(session.chunks)
            return doc

    # ---- reading ---------------------------------------------------------

    def documents(self, session_id: str | None) -> list[Document]:
        if not session_id:
            return []
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return []
            s.touched = time.monotonic()
            return list(s.documents)

    def passage_counts(self, session_id: str | None) -> dict[str, int]:
        """Passages per uploaded paper, for the library listing."""
        if not session_id:
            return {}
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return {}
            counts: dict[str, int] = {}
            for c in s.chunks:
                counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
            return counts

    def search(
        self,
        session_id: str | None,
        query: str,
        config: RetrievalConfig,
        doc_ids: set[str] | None = None,
    ) -> list[Retrieved]:
        """Rank this session's uploaded passages. Empty when there are none."""
        if not session_id:
            return []
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None or not s.chunks:
                return []
            s.touched = time.monotonic()
            pipeline = s.pipeline
            mine = s.doc_ids
        if doc_ids is not None and not (doc_ids & mine):
            # The reader chose papers and none of them is theirs.
            return []
        results = pipeline.search(query, config, doc_ids)
        return [
            Retrieved(
                chunk=r.chunk, score=r.score, dense_score=r.dense_score,
                bm25_score=r.bm25_score, rrf_score=r.rrf_score,
                rerank_score=r.rerank_score,
                sources=r.sources | {"upload"},
            )
            for r in results
        ]
