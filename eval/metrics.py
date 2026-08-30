"""Retrieval metrics.

Deliberately dependency-free and written out longhand. These four numbers are
the project's entire claim to being measured rather than demonstrated, so they
are the last place to take a library on trust — and each is short enough that
the implementation is easier to check than a call signature would be.

Relevance is binary throughout: a chunk either was or was not labelled as
supporting the answer. Graded relevance would be more informative and is not
worth the labelling cost at 60 questions, where the marginal judgement between
"quite relevant" and "very relevant" is mostly annotator noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the labelled passages that appear in the top k.

    The headline number. If the right passage is not in the candidate set, no
    amount of generation quality recovers the answer, so this bounds everything
    downstream.
    """
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k that was labelled relevant.

    Reported but not optimised for. With 1-3 labelled passages per question,
    precision@10 is capped near 0.3 by construction, so its absolute value says
    more about the labelling than the retriever. It is useful only as a
    relative signal between configurations.
    """
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: set[str], k: int | None = None) -> float:
    """1 / rank of the first relevant result; 0 if none is found.

    Sensitive to exactly what matters when the context window is small: not
    whether the right passage was found, but whether it was found *first*.
    """
    limit = len(retrieved) if k is None else k
    for i, chunk_id in enumerate(retrieved[:limit], start=1):
        if chunk_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain over binary relevance.

    Unlike recall, this is sensitive to the ordering *within* the top k, which
    is the only thing a reranker changes. A reranker that improves nDCG while
    leaving recall flat has done its job — it did not find anything new, it put
    what was already there in a better order.
    """
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in relevant
    )
    # Ideal: every relevant passage packed into the top positions.
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """One configuration's scores over the whole question set."""

    config: str
    n_questions: int
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    mrr: float
    ndcg_at_10: float
    #: Median wall-clock per query, milliseconds. A retrieval improvement that
    #: costs 400ms is a different trade than one that costs 8ms, and the
    #: ablation table should let a reader see which they are looking at.
    median_latency_ms: float

    def as_row(self) -> str:
        return (
            f"| {self.config:<22} | {self.recall_at_5:>8.3f} | {self.recall_at_10:>9.3f} "
            f"| {self.mrr:>6.3f} | {self.ndcg_at_10:>8.3f} | {self.median_latency_ms:>7.0f} |"
        )

    @staticmethod
    def header() -> str:
        return (
            "| configuration          | Recall@5 | Recall@10 |    MRR | nDCG@10 | ms/q    |\n"
            "|------------------------|----------|-----------|--------|---------|---------|"
        )


def aggregate(
    config: str,
    per_question: list[tuple[list[str], set[str]]],
    latencies_ms: list[float],
) -> RetrievalScore:
    """Mean each metric over questions.

    Macro-averaged: every question counts once, regardless of how many
    passages were labelled for it. Micro-averaging would let a question with
    five labelled passages outweigh four questions with one each, which is an
    artefact of labelling effort rather than of retrieval quality.
    """
    n = len(per_question)
    if n == 0:
        raise ValueError("no questions to score — check the ground-truth file loaded")

    def mean(f) -> float:
        return sum(f(r, rel) for r, rel in per_question) / n

    lat = sorted(latencies_ms)
    median = lat[len(lat) // 2] if lat else 0.0

    return RetrievalScore(
        config=config,
        n_questions=n,
        recall_at_5=mean(lambda r, rel: recall_at_k(r, rel, 5)),
        recall_at_10=mean(lambda r, rel: recall_at_k(r, rel, 10)),
        precision_at_5=mean(lambda r, rel: precision_at_k(r, rel, 5)),
        mrr=mean(lambda r, rel: reciprocal_rank(r, rel, 10)),
        ndcg_at_10=mean(lambda r, rel: ndcg_at_k(r, rel, 10)),
        median_latency_ms=median,
    )
