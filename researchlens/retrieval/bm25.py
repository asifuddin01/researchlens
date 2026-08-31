"""Lexical retrieval.

The half of hybrid retrieval that finds "Dice 0.91", "SwinUNETR", "AdamW" and
"KiTS23" — exact strings a dense encoder maps into a neighbourhood of things
that merely mean something similar, which for a metric value or a dataset
identifier is precisely wrong.

Okapi BM25, implemented here rather than taken from a library. It is forty
lines, the ablation table rests on it, and a dependency would have to be
trusted on exactly the tokenisation decisions that matter most for scientific
text — see `tokenise` for what those are.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from researchlens.types import Chunk

#: Keeps hyphenated names, decimals and alphanumeric model names whole:
#: "U-Net", "0.94", "bge-small", "3D", "scGPT". Splitting any of those is the
#: difference between finding the paper and finding papers about the topic.
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")

#: Words too common to discriminate. Deliberately short: an aggressive list
#: removes "not", which changes the meaning of a query about negative results,
#: and removes "all" from "attention is all you need".
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "these this to was were will with".split()
)


def tokenise(text: str) -> list[str]:
    """Lowercase, keep the characters that carry identity in scientific text.

    No stemming. "segmenting" and "segmentation" staying distinct costs little,
    while a stemmer reliably mangles model names — Porter turns "GEARS" into
    "gear" and "scGPT" into "scgpt" only by accident of casing. The whole point
    of the lexical half is exactness.
    """
    return [t for t in (m.group(0).lower() for m in _TOKEN.finditer(text)) if t not in _STOP]


class BM25Retriever:
    """Okapi BM25 over the chunk collection."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        #: Term-frequency saturation. Above k1 occurrences a term stops adding
        #: much, which stops a passage that repeats "perturbation" twenty times
        #: from beating one that uses it twice and answers the question.
        self.k1 = k1
        #: Length normalisation. 0.75 is the standard compromise; at 1.0 short
        #: passages dominate, at 0 long ones do.
        self.b = b
        self._ids: list[str] = []
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        self._avg_len = 0.0
        self._idf: dict[str, float] = {}

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("nothing to index — the corpus produced no chunks")

        self._ids = [c.chunk_id for c in chunks]
        self._tf = [Counter(tokenise(c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = sum(self._len) / len(self._len)

        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())

        n = len(self._tf)
        # Robertson/Sparck Jones idf with the +0.5 smoothing, floored at zero.
        # Unfloored, a term appearing in more than half the corpus scores
        # negative and actively pushes down passages that contain it.
        self._idf = {
            term: max(0.0, math.log((n - d + 0.5) / (d + 0.5) + 1.0)) for term, d in df.items()
        }

    def search(
        self, query: str, k: int, allow: set[int] | None = None
    ) -> list[tuple[str, float]]:
        """Rank chunks against the query, optionally within a subset.

        `allow` holds row indices, not ids, because this is the inner loop.

        Restricting here rather than filtering the results afterwards is not an
        optimisation. Filtering after the fact returns nothing when scores tie:
        the tie-break is alphabetical by id, so a wide pool fills with
        alphabetically-early papers and the selected one never appears in it. A
        reader who picked one paper would be told the corpus had nothing.
        """
        if not self._ids:
            raise RuntimeError("search() before index()")

        terms = tokenise(query)
        scores: dict[int, float] = {}

        for term in terms:
            idf = self._idf.get(term)
            if not idf:
                continue
            for i, tf in enumerate(self._tf):
                if allow is not None and i not in allow:
                    continue
                f = tf.get(term)
                if not f:
                    continue
                norm = 1 - self.b + self.b * (self._len[i] / self._avg_len)
                scores[i] = scores.get(i, 0.0) + idf * (f * (self.k1 + 1)) / (f + self.k1 * norm)

        # Ties break on chunk_id so two runs over one index agree exactly;
        # without it, dict order leaks into the published metrics.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self._ids[kv[0]]))
        return [(self._ids[i], s) for i, s in ranked[:k]]

    def __len__(self) -> int:
        return len(self._ids)
