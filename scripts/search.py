"""Query the corpus from the command line.

    python scripts/search.py "how is a gene regulatory network inferred"

Exists to make retrieval inspectable before there is any generation to hide
behind. Reading what actually comes back for a query is the fastest way to tell
a chunking problem from a retrieval problem, and it needs no ground truth.

A caution about labelling: do not build `eval/questions.jsonl` by running a
query here and marking the top hits relevant. That labels the system correct by
construction and produces an ablation in which every configuration scores well.
Write the question, find the passage by reading the paper, then record it.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Runnable as `python scripts/search.py` without an editable install, so that
# inspecting retrieval never depends on having set the project up correctly —
# it is the tool you reach for when something else is wrong.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from researchlens.ingest.chunk import chunk_corpus  # noqa: E402
from researchlens.ingest.library import load_library  # noqa: E402
from researchlens.retrieval.dense import DenseRetriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Search the indexed corpus.")
    ap.add_argument("query", nargs="+")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--chars", type=int, default=280, help="passage preview length")
    args = ap.parse_args()
    query = " ".join(args.query)

    docs, skipped = load_library(ROOT / "data" / "pdfs", ROOT / "data" / "index")
    for m in skipped:
        print(f"  skipped: {m}", file=sys.stderr)
    chunks = chunk_corpus(docs)

    retriever = DenseRetriever()
    retriever.index(chunks)
    by_id = {c.chunk_id: c for c in chunks}

    print(f"\n{len(docs)} papers · {len(chunks)} passages\n")
    print(f'  "{query}"\n')
    for rank, (chunk_id, score) in enumerate(retriever.search(query, args.k), start=1):
        c = by_id[chunk_id]
        print(f"  [{rank}] {score:.3f}  {c.doc_title[:66]}")
        print(f"       {c.section_heading[:48]}  ·  p{c.pages}  ·  {c.section_kind}")
        text = " ".join(c.text.split())
        print(f"       {text[:args.chars]}...\n")


if __name__ == "__main__":
    main()
