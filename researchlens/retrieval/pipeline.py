"""The retrieval pipeline, configurable along the ablation axis.

This module exists so that the ablation table is *generated*. The four rows in
the README are not four code paths that were each written and measured once —
they are one code path run under four `RetrievalConfig` values. A row that
cannot be produced by flipping a flag here is a row that does not go in the
table.

That constraint is the point. It makes the comparison honest (every
configuration shares the parser, chunker, index and query handling, so the only
difference is the one named), and it makes it cheap to rerun after a change,
which is what stops the published numbers drifting away from the code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from researchlens.retrieval.fusion import reciprocal_rank_fusion
from researchlens.types import Chunk, Retrieved


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """One point in the ablation.

    `label` is what appears in the table's first column, so it is part of the
    published output rather than an internal name.
    """

    label: str
    use_dense: bool = True
    use_bm25: bool = False
    use_rerank: bool = False
    #: Candidates pulled from each retriever before fusion. Wider than the
    #: final k because fusion can only reorder what it was given, and the
    #: reranker needs room to move something up from rank 20.
    candidates: int = 30
    #: Passages actually returned.
    top_k: int = 10

    def __post_init__(self) -> None:
        if not (self.use_dense or self.use_bm25):
            raise ValueError(
                f"{self.label}: a configuration with neither dense nor BM25 "
                "retrieval has nothing to rank"
            )


#: The published ablation. Each row adds exactly one component to the row above
#: it, so the delta between adjacent rows is attributable to that component and
#: nothing else. Reordering or skipping a step here breaks that reading.
ABLATION: list[RetrievalConfig] = [
    RetrievalConfig("dense only", use_dense=True, use_bm25=False),
    RetrievalConfig("bm25 only", use_dense=False, use_bm25=True),
    RetrievalConfig("hybrid (RRF)", use_dense=True, use_bm25=True),
    RetrievalConfig("hybrid + rerank", use_dense=True, use_bm25=True, use_rerank=True),
]


class RetrievalPipeline:
    """Composes retrievers according to a config.

    Holds no model state of its own — the retrievers are injected, so the
    ablation loop builds and indexes each retriever once and then runs all four
    configurations against them. Rebuilding the index per configuration would
    quadruple the run and, worse, would let an index difference masquerade as a
    retrieval difference.
    """

    def __init__(self, dense=None, bm25=None, reranker=None) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker
        self._by_id: dict[str, Chunk] = {}
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk], skip_dense: bool = False) -> None:
        # The list is kept, not just the map. Row indices passed to the
        # retrievers are positions in the sequence they indexed, and deriving
        # them from a dict's insertion order would couple correctness to a
        # detail that holds today and says nothing about tomorrow. A duplicate
        # id would silently shift every index after it and retrieve the wrong
        # passage confidently.
        self._chunks = list(chunks)
        self._by_id = {c.chunk_id: c for c in chunks}
        if len(self._by_id) != len(self._chunks):
            raise ValueError(
                f"{len(self._chunks) - len(self._by_id)} duplicate chunk ids — "
                "row indices would no longer match the retrievers"
            )
        if self.dense is not None and not skip_dense:
            self.dense.index(chunks)
        if self.bm25 is not None:
            self.bm25.index(chunks)

    def index_without_dense(self, chunks: list[Chunk]) -> None:
        """Index everything except the dense vectors.

        For a deployment loading a prebuilt bundle: the embeddings are already
        computed, and re-embedding 9,870 passages at every container start
        would make a scale-to-zero demo unusable.
        """
        self.index(chunks, skip_dense=True)

    def search(
        self,
        query: str,
        config: RetrievalConfig,
        doc_ids: set[str] | None = None,
    ) -> list[Retrieved]:
        """Run one query under one configuration, optionally within a subset.

        `doc_ids` restricts the answer to chosen papers. Filtering happens
        after retrieval rather than before, because the BM25 statistics and the
        dense matrix are built over the whole corpus and rebuilding either per
        request would cost more than the query.

        That makes the candidate pool the thing to get right: asking two papers
        out of a hundred, the top thirty results are unlikely to contain thirty
        passages from those two. The pool therefore widens with how narrow the
        selection is, so a specific question about a specific paper does not
        come back empty because the corpus had more to say elsewhere.
        """
        if config.use_dense and self.dense is None:
            raise RuntimeError(f"{config.label} needs a dense retriever; none was given")
        if config.use_bm25 and self.bm25 is None:
            raise RuntimeError(f"{config.label} needs a BM25 retriever; none was given")
        if config.use_rerank and self.reranker is None:
            raise RuntimeError(f"{config.label} needs a reranker; none was given")

        allow = self._rows_for(doc_ids)
        if allow is not None and not allow:
            return []

        rankings: dict[str, list[str]] = {}
        raw: dict[str, dict[str, float]] = {}

        if config.use_dense:
            hits = self.dense.search(query, config.candidates, allow)
            rankings["dense"] = [cid for cid, _ in hits]
            raw["dense"] = dict(hits)
        if config.use_bm25:
            hits = self.bm25.search(query, config.candidates, allow)
            rankings["bm25"] = [cid for cid, _ in hits]
            raw["bm25"] = dict(hits)

        # With one retriever, fusion is an identity on the ordering. Running it
        # anyway keeps every configuration on the same code path, so a bug in
        # fusion cannot hide in the rows that do not use it.
        fused = reciprocal_rank_fusion(rankings)

        if config.use_rerank:
            candidates = [self._by_id[cid] for cid, _, _ in fused[: config.candidates]]
            reranked = self.reranker.rerank(query, candidates, config.top_k)
            order = {cid: score for cid, score in reranked}
            rrf = {cid: s for cid, s, _ in fused}
            srcs = {cid: src for cid, _, src in fused}
            return [
                Retrieved(
                    chunk=self._by_id[cid],
                    score=score,
                    bm25_score=raw.get("bm25", {}).get(cid),
                    dense_score=raw.get("dense", {}).get(cid),
                    rrf_score=rrf.get(cid),
                    rerank_score=score,
                    sources=srcs.get(cid, frozenset()),
                )
                for cid, score in reranked[: config.top_k]
            ]

        return [
            Retrieved(
                chunk=self._by_id[cid],
                score=score,
                bm25_score=raw.get("bm25", {}).get(cid),
                dense_score=raw.get("dense", {}).get(cid),
                rrf_score=score,
                sources=srcs,
            )
            for cid, score, srcs in fused[: config.top_k]
        ]

    def _rows_for(self, doc_ids: set[str] | None) -> set[int] | None:
        """Row indices belonging to the chosen papers.

        Both retrievers index chunks in the order they were given, so one
        ordering serves both. Computed per call rather than cached: a reader
        changes the selection far more often than the corpus changes, and the
        pass is over ids rather than text.
        """
        if doc_ids is None:
            return None
        return {i for i, c in enumerate(self._chunks) if c.doc_id in doc_ids}

    def timed_search(
        self, query: str, config: RetrievalConfig, doc_ids: set[str] | None = None
    ) -> tuple[list[Retrieved], float]:
        """Search, returning elapsed milliseconds alongside the results.

        Latency belongs in the ablation table. A component that buys three
        points of recall for 8ms and one that buys three points for 400ms are
        different engineering decisions, and a table that reports only quality
        hides which one is on offer.
        """
        start = time.perf_counter()
        results = self.search(query, config, doc_ids)
        return results, (time.perf_counter() - start) * 1000.0
