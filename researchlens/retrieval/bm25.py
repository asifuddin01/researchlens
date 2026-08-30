"""Lexical retrieval.

Phase II. The half of hybrid retrieval that finds "Dice 0.91", "SwinUNETR",
"KiTS23" and "external validation" — exact strings that a dense encoder maps
into a neighbourhood of things that merely mean something similar, which for a
metric name or a dataset identifier is precisely wrong.

Planned implementation: `rank_bm25`'s Okapi variant over a light analyser
(lowercase, strip punctuation, keep digits and hyphenated tokens intact so
"U-Net" and "3D" survive). No stemming — "segmenting" and "segmentation" being
distinct costs little here, while stemming reliably damages model names.
"""

from __future__ import annotations

from researchlens.types import Chunk


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError("Phase II — see docs/phases.md")

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        raise NotImplementedError("Phase II — see docs/phases.md")
