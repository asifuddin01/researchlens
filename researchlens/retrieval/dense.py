"""Dense retrieval over ONNX sentence embeddings.

The first retriever implemented, because the baseline that every later number
is measured against is dense-only. Until this works there is nothing to compare
BM25 or reranking to.

Model: BAAI/bge-small-en-v1.5 — 384 dimensions, 67 MB. Small is a requirement
rather than a compromise: the retrieval service has to stay under the memory
ceiling that lets a scale-to-zero host snapshot and resume it, and a larger
encoder would push it into cold-boot territory for a recall gain that has not
been shown to exist on this corpus. `bge-base-en-v1.5` is a 210 MB drop-in if
the ablation ever says the gain is real.

Vectors are stored in one dense numpy matrix and searched exhaustively. At the
few thousand chunks this corpus produces, an approximate index would be slower
to build, no faster to query, and would introduce a recall loss indistinguishable
from a retrieval bug. The seam for scale is `search`: past ~10^5 chunks, swap
the matrix product for HNSW behind this same method.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

from researchlens.types import Chunk

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

#: BGE's own guidance suggests prefixing *queries* (not passages) with a short
#: instruction for short-query retrieval, while noting v1.5 retrieves well
#: without it. fastembed applies no prefix for this model — `query_embed`,
#: `passage_embed` and `embed` were verified to return identical vectors — so
#: nothing is being added behind our back.
#:
#: Left off by default. It is a lever whose effect is unknown on this corpus,
#: and the entire discipline of this project is that a lever gets pulled when
#: there is ground truth to measure it against, not before.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    """Embeds chunks once, then answers queries by exhaustive cosine search."""

    name = "dense"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        query_instruction: str | None = None,
        cache_dir: Path | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model = model
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.cache_dir = cache_dir or Path(
            os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent.parent / "data")
        ) / "index"
        self._encoder = None
        self._ids: list[str] = []
        #: (n_chunks, dim), float32, L2-normalised — so cosine similarity is a
        #: plain dot product and the search is one matrix-vector multiply.
        self._matrix: np.ndarray | None = None

    # ---- model -----------------------------------------------------------

    def _lazy_encoder(self):
        """Load the encoder on first use.

        Deferred because constructing a `DenseRetriever` happens in the
        ablation setup for every configuration, including the BM25-only row
        that never touches it. Loading ONNX weights there would add seconds to
        a run that does not use them.
        """
        if self._encoder is None:
            from fastembed import TextEmbedding

            self._encoder = TextEmbedding(model_name=self.model)
        return self._encoder

    # ---- cache -----------------------------------------------------------

    def _cache_key(self, chunks: list[Chunk]) -> str:
        """Identify this exact corpus under this exact model.

        Hashes the chunk ids *and* their text. Ids alone would collide when the
        chunker is re-tuned but happens to produce the same count, and a stale
        cache silently scoring against the wrong passages is the kind of bug
        that survives a whole evening.
        """
        h = hashlib.sha256()
        h.update(self.model.encode())
        h.update(str(self.query_instruction).encode())
        for c in chunks:
            h.update(c.chunk_id.encode())
            h.update(c.text.encode())
        return h.hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"dense-{key}.npz"

    # ---- indexing --------------------------------------------------------

    def index(self, chunks: list[Chunk]) -> None:
        """Embed every chunk, or load the embeddings from cache."""
        if not chunks:
            raise ValueError("nothing to index — the corpus produced no chunks")

        self._ids = [c.chunk_id for c in chunks]
        key = self._cache_key(chunks)
        path = self._cache_path(key)

        if path.exists():
            with np.load(path) as z:
                self._matrix = z["matrix"]
            print(f"  dense: {len(self._ids)} vectors from cache", file=sys.stderr)
            return

        encoder = self._lazy_encoder()
        texts = [c.text for c in chunks]
        vectors: list[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(np.asarray(v, dtype=np.float32) for v in encoder.embed(batch))
            print(
                f"\r  dense: embedding {min(start + len(batch), len(texts))}/{len(texts)}",
                end="",
                file=sys.stderr,
            )
        print(file=sys.stderr)

        matrix = np.vstack(vectors).astype(np.float32)
        # fastembed returns unit vectors for this model, but normalising is
        # cheap and makes the dot-product-is-cosine assumption in `search` true
        # by construction rather than by trust in an upstream default.
        matrix = _l2_normalise(matrix)
        self._matrix = matrix

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, matrix=matrix)

    # ---- search ----------------------------------------------------------

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return the k nearest chunks as `(chunk_id, cosine)`, descending."""
        if self._matrix is None:
            raise RuntimeError("search() before index()")

        text = f"{self.query_instruction}{query}" if self.query_instruction else query
        vec = np.asarray(next(iter(self._lazy_encoder().embed([text]))), dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) or 1.0)

        scores = self._matrix @ vec

        k = min(k, len(self._ids))
        if k <= 0:
            return []
        # argpartition finds the top k without sorting the rest; only those k
        # are then ordered. Irrelevant at 3,000 chunks, correct at 300,000.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self._ids)


def _l2_normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    # A zero vector would divide by zero. It should not occur — an empty chunk
    # cannot reach here — but a silent NaN would propagate into every score and
    # present as uniformly broken retrieval rather than as one bad chunk.
    norms[norms == 0] = 1.0
    return m / norms
