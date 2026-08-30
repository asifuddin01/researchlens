"""Cross-encoder reranking.

Phase II. A bi-encoder embeds query and passage separately, so it never sees
them together; a cross-encoder scores the pair jointly and can notice that a
passage mentions the right metric on the wrong dataset. The cost is that it
cannot be precomputed — every candidate is a forward pass — which is why it
runs over ~30 fused candidates and not over the corpus.

Planned implementation: `fastembed`'s TextCrossEncoder with
BAAI/bge-reranker-base, ONNX, ~280 MB quantised.

Worth stating before the numbers exist: this may not help. On a 25-paper corpus
where hybrid retrieval already puts the labelled passage in the top 5, there is
little room above it. If the ablation says so, that row goes in the README
unchanged — it is the only result in this project that could be surprising.
"""

from __future__ import annotations

from researchlens.types import Chunk

DEFAULT_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderReranker:
    name = "rerank"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def rerank(
        self, query: str, candidates: list[Chunk], top_k: int
    ) -> list[tuple[str, float]]:
        raise NotImplementedError("Phase II — see docs/phases.md")
