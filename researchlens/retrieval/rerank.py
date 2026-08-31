"""Cross-encoder reranking.

A bi-encoder embeds query and passage separately and so never sees them
together; a cross-encoder scores the pair jointly and can notice that a passage
reports the right metric on the wrong dataset. The cost is that nothing can be
precomputed — every candidate is a forward pass — which is why it runs over the
~30 fused candidates and never over the corpus.

Stated before the numbers exist: this may not help. On a corpus where hybrid
retrieval already puts the labelled passage in the top five there is little
room above it. If the ablation says so, that row goes in the README unchanged.
It is the only result in this project that could be surprising.
"""

from __future__ import annotations

import os

from researchlens.types import Chunk

DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reorders fused candidates by scoring each against the query jointly."""

    name = "rerank"

    def __init__(self, model: str = DEFAULT_MODEL, threads: int | None = None) -> None:
        self.model = model
        self.threads = threads if threads is not None else (os.cpu_count() or 4)
        self._encoder = None

    def _lazy_encoder(self):
        """Load on first use.

        A reranker is constructed for any ablation run that includes the
        reranking row, but the rows before it never call `rerank`. Loading the
        weights in the constructor would add seconds to configurations that do
        not use them, and the ablation's latency column would then measure the
        harness rather than the retriever.
        """
        if self._encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            # CPU, deliberately, and *not* the CoreML provider the embedder
            # uses. Measured on this machine, 30 real passages of ~900 chars:
            #
            #   CoreML             12408 ms
            #   CPU                  507 ms
            #   CPU, threads=8       471 ms
            #
            # CoreML is 24x slower here while being 2.6x faster for the
            # bi-encoder. onnxruntime reports why: of 327 graph nodes it can
            # place 212, split across 39 partitions, so every forward pass
            # thrashes between CPU and the neural engine. The lesson is that a
            # provider is a property of the model, not of the machine — this
            # cost 111 seconds per query before it was measured.
            self._encoder = TextCrossEncoder(model_name=self.model, threads=self.threads)
        return self._encoder

    def rerank(
        self, query: str, candidates: list[Chunk], top_k: int
    ) -> list[tuple[str, float]]:
        """Score each candidate against the query, best first.

        Scores are raw cross-encoder logits, not probabilities: they are
        comparable within one query and meaningless across queries. Nothing
        downstream compares them across queries, and squashing them through a
        sigmoid would only make that misuse look reasonable.
        """
        if not candidates:
            return []

        scores = list(self._lazy_encoder().rerank(query, [c.text for c in candidates]))
        paired = list(zip((c.chunk_id for c in candidates), scores, strict=True))
        # Ties break on chunk_id, so a rerun reproduces the metrics exactly.
        paired.sort(key=lambda t: (-t[1], t[0]))
        return [(cid, float(s)) for cid, s in paired[:top_k]]
