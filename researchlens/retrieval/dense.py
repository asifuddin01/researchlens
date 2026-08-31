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


def _select_providers() -> list[str] | None:
    """Prefer CoreML on Apple silicon, fall back everywhere else.

    Measured on this corpus (192 real passages, bge-small, 8 cores):

        default                 2.2/s
        threads=8               4.0/s
        CoreML                  5.7/s
        threads=8, parallel=0   1.2/s

    Two things worth recording. CoreML is the clear winner where it exists, so
    it is requested by name — but it is macOS-only and the container runs
    Linux, so it is detected rather than assumed. And fastembed's `parallel`
    multiprocessing is *slower*, not faster: each worker loads its own copy of
    the model, and at this corpus size the reload dominates. It is deliberately
    not used.
    """
    try:
        import onnxruntime as ort

        if "CoreMLExecutionProvider" in ort.get_available_providers():
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return None  # fastembed picks its own default


class DenseRetriever:
    """Embeds chunks once, then answers queries by exhaustive cosine search."""

    name = "dense"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        query_instruction: str | None = None,
        cache_dir: Path | None = None,
        batch_size: int = 64,
        threads: int | None = None,
    ) -> None:
        self.threads = threads if threads is not None else (os.cpu_count() or 4)
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

            kwargs: dict = {"model_name": self.model, "threads": self.threads}
            providers = _select_providers()
            if providers:
                kwargs["providers"] = providers
            self._encoder = TextEmbedding(**kwargs)
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

        # One call, not a Python-level loop over batches: fastembed batches
        # internally and re-entering it per batch only adds setup. Progress is
        # reported from the generator so a ten-minute first run is visibly
        # working rather than apparently hung.
        for i, v in enumerate(encoder.embed(texts, batch_size=self.batch_size), start=1):
            vectors.append(np.asarray(v, dtype=np.float32))
            if i % 64 == 0 or i == len(texts):
                print(f"\r  dense: embedding {i}/{len(texts)}", end="", file=sys.stderr)
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

    def search(
        self, query: str, k: int, allow: set[int] | None = None
    ) -> list[tuple[str, float]]:
        """Return the k nearest chunks as `(chunk_id, cosine)`, descending.

        `allow` holds row indices. Masking the score vector costs one pass and
        guarantees the subset is represented; filtering afterwards does not,
        because the top k over the whole corpus may contain none of it.
        """
        if self._matrix is None:
            raise RuntimeError("search() before index()")

        text = f"{self.query_instruction}{query}" if self.query_instruction else query
        vec = np.asarray(next(iter(self._lazy_encoder().embed([text]))), dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) or 1.0)

        scores = self._matrix @ vec

        if allow is not None:
            # -inf rather than 0: a cosine can legitimately be negative, and
            # zeroing would rank an excluded passage above a poorly-matching
            # included one.
            mask = np.full(scores.shape, -np.inf, dtype=scores.dtype)
            idx = np.fromiter(allow, dtype=np.int64, count=len(allow))
            mask[idx] = scores[idx]
            scores = mask
            k = min(k, len(allow))

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
