"""The evaluation harness.

Run before writing retrieval code, not after. Its job is to make every later
change arrive with a number attached, which is the only thing separating this
project from a demo that looks convincing.

    python -m eval.run --config "dense only"    # one configuration
    python -m eval.run --ablation               # every row, as a markdown table

The output of `--ablation` is pasted into the README by `make ablation`. It is
never edited by hand: a table that can be edited is a table that will drift
away from the code, and at that point it is decoration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.metrics import RetrievalScore, aggregate
from researchlens.ingest.chunk import chunk_corpus, fingerprint
from researchlens.ingest.library import load_library
from researchlens.retrieval.pipeline import ABLATION, RetrievalConfig, RetrievalPipeline
from researchlens.types import Chunk, Document

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
INDEX_DIR = ROOT / "data" / "index"
QUESTIONS = ROOT / "eval" / "questions.jsonl"


def load_questions(path: Path = QUESTIONS) -> tuple[list[dict], str]:
    """Read the hand-labelled ground truth.

    Returns the rows and the chunking fingerprint they were written under. The
    fingerprint is stored in a header row rather than inferred, because the
    whole point is to detect that the chunker changed *after* the labels were
    written — which cannot be detected from the labels themselves.
    """
    if not path.exists():
        sys.exit(
            f"No ground truth at {path}.\n"
            "This file is written by hand and is the deliverable that makes every\n"
            "other number in this project real. See eval/README.md for the format."
        )

    rows: list[dict] = []
    fp: str | None = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        obj = json.loads(line)
        if obj.get("_meta"):
            fp = obj.get("chunking")
            continue
        rows.append(obj)

    if fp is None:
        sys.exit(f"{path} has no _meta row recording the chunking fingerprint.")
    return rows, fp


def load_corpus() -> tuple[list[Document], list[Chunk]]:
    """Load the parsed library, then chunk it.

    Parsing goes through the on-disk cache rather than happening here, so every
    configuration in an ablation sees byte-identical documents. Re-parsing
    between rows would let an extraction difference appear as a retrieval one.
    """
    try:
        docs, skipped = load_library(PDF_DIR, INDEX_DIR)
    except FileNotFoundError:
        sys.exit(
            f"No PDFs in {PDF_DIR}.\n"
            "Fetch the corpus first:  python scripts/fetch_corpus.py\n"
            "Or point it at your own folder:  make ingest DIR=/path/to/papers"
        )

    for msg in skipped:
        print(f"  skipped: {msg}", file=sys.stderr)
    if not docs:
        sys.exit("Every PDF failed to parse — nothing to evaluate.")

    return docs, chunk_corpus(docs)


def build_pipeline(chunks: list[Chunk], need_rerank: bool) -> RetrievalPipeline:
    """Build and index every retriever once, for all configurations.

    Indexing per configuration would let an index difference masquerade as a
    retrieval difference, which is the one thing an ablation must not permit.
    """
    from researchlens.retrieval.bm25 import BM25Retriever
    from researchlens.retrieval.dense import DenseRetriever
    from researchlens.retrieval.rerank import CrossEncoderReranker

    pipe = RetrievalPipeline(
        dense=DenseRetriever(),
        bm25=BM25Retriever(),
        reranker=CrossEncoderReranker() if need_rerank else None,
    )
    pipe.index(chunks)
    return pipe


def score_config(
    pipe: RetrievalPipeline, config: RetrievalConfig, questions: list[dict]
) -> RetrievalScore:
    per_question: list[tuple[list[str], set[str]]] = []
    latencies: list[float] = []

    for q in questions:
        results, ms = pipe.timed_search(q["question"], config)
        per_question.append(([r.chunk.chunk_id for r in results], set(q["relevant_chunks"])))
        latencies.append(ms)

    return aggregate(config.label, per_question, latencies)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="run a single configuration by label")
    ap.add_argument("--ablation", action="store_true", help="run every configuration")
    args = ap.parse_args()

    questions, label_fp = load_questions()
    docs, chunks = load_corpus()
    current_fp = fingerprint()

    if label_fp != current_fp:
        sys.exit(
            f"Chunking changed since the labels were written.\n"
            f"  labels written under: {label_fp}\n"
            f"  current parameters:   {current_fp}\n"
            "Chunk ids no longer mean what the labels say they mean. Either revert\n"
            "the chunking parameters, or re-label. Scoring anyway would produce a\n"
            "confidently wrong number, which is worse than no number."
        )

    print(
        f"{len(docs)} papers · {len(chunks)} chunks · {len(questions)} questions "
        f"· chunking {current_fp}\n",
        file=sys.stderr,
    )

    configs = ABLATION if args.ablation else [
        c for c in ABLATION if c.label == args.config
    ] or [ABLATION[0]]

    need_rerank = any(c.use_rerank for c in configs)
    pipe = build_pipeline(chunks, need_rerank)

    print(RetrievalScore.header())
    for config in configs:
        print(score_config(pipe, config, questions).as_row(), flush=True)


if __name__ == "__main__":
    main()
