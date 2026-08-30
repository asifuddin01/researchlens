"""Dense retrieval.

Phase I needs this, because the baseline the whole project measures against is
dense-only. It is therefore the one retriever that must work before anything
else does.

Planned implementation: `fastembed` (ONNX, Apache-2.0) with BAAI/bge-small-en-v1.5
— ~130 MB, which is what keeps the retrieval service under the 2 GB ceiling that
lets Fly suspend it. Vectors held in a plain numpy array with cosine similarity:
at ~3,000 chunks a vector database is complexity theatre, and an exact search is
both faster than an approximate one at this size and immune to a recall loss
that would be indistinguishable from a retrieval bug.

The seam for scale is `search`. Past ~10^5 chunks, swap the numpy scan for an
HNSW index behind this same method.
"""

from __future__ import annotations

from researchlens.types import Chunk

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class DenseRetriever:
    name = "dense"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError("Phase I — the baseline depends on this one")

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        raise NotImplementedError("Phase I — the baseline depends on this one")
