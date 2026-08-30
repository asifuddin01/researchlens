"""Reciprocal rank fusion.

RRF combines rankings by position rather than by score, which is what makes it
usable here: BM25 returns unbounded positive scores and cosine similarity
returns numbers in [-1, 1], and no fixed normalisation of the two survives a
change of corpus. Positions are directly comparable and need no tuning.

Cormack et al. (2009) showed RRF beating the best individual system and several
score-normalising fusions across TREC runs, without a trained combiner. That
"without a trained combiner" is the point at 60 questions: anything with a
weight to fit would be fitted on the same data it is evaluated against.
"""

from __future__ import annotations

from collections import defaultdict

#: Cormack's constant. It damps the influence of the very top ranks, so a
#: single retriever cannot dominate the fused ordering from position 1 alone.
#: Left at the published value deliberately — tuning it on 60 questions would
#: fit noise, and any gain would not survive a new corpus.
DEFAULT_K = 60


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    k: int = DEFAULT_K,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float, frozenset[str]]]:
    """Fuse several ranked id lists into one.

    Returns `(chunk_id, score, sources)` descending by score, where `sources`
    names which retrievers proposed the id. That third element is not
    decoration: "found by BM25, missed by dense" is the observation that tells
    you *why* hybrid beat either, and it is the difference between reporting an
    ablation and explaining one.

    Ties break on chunk_id so that two runs over the same index produce
    identical output. Without it, dict iteration order leaks into the metrics
    and a rerun shows a different Recall@5 for no reason.
    """
    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, set[str]] = defaultdict(set)

    for name, ranked in rankings.items():
        w = 1.0 if weights is None else weights.get(name, 1.0)
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] += w / (k + rank)
            sources[chunk_id].add(name)

    return sorted(
        ((cid, s, frozenset(sources[cid])) for cid, s in scores.items()),
        key=lambda t: (-t[1], t[0]),
    )
