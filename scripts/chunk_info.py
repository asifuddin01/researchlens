"""Report the exact chunker configuration and what it produces.

Written for whoever is about to hand-label `eval/questions.jsonl`: the labels
name chunk ids, and a chunk id only means anything under the parameters that
produced it. The harness refuses to score if the fingerprint printed here does
not match the one recorded in the ground-truth file.
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from researchlens.ingest import chunk as C  # noqa: E402
from researchlens.ingest.library import PARSER_VERSION, load_library  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    docs, skipped = load_library(ROOT / "data" / "pdfs", ROOT / "data" / "index")
    chunks = C.chunk_corpus(docs)
    lengths = np.array([len(c.text) for c in chunks])
    per_doc = Counter(c.doc_id for c in chunks)

    print(f"\n  fingerprint        {C.fingerprint()}")
    print(f"  chunk size         {C.DEFAULT_SIZE} characters")
    print(f"  overlap            {C.DEFAULT_OVERLAP} characters")
    print(f"  parser version     v{PARSER_VERSION}")
    print(f"\n  papers indexed     {len(docs)}  (skipped {len(skipped)})")
    print(f"  sections           {sum(len(d.sections) for d in docs)}")
    print(f"  CHUNKS             {len(chunks)}")
    print(
        f"\n  chunk length       min {lengths.min()} · p50 {int(np.percentile(lengths, 50))}"
        f" · p95 {int(np.percentile(lengths, 95))} · max {lengths.max()}"
        f" · mean {lengths.mean():.0f}"
    )
    print(f"  total characters   {lengths.sum():,}")
    print(
        f"  chunks per paper   min {min(per_doc.values())}"
        f" · median {int(np.median(list(per_doc.values())))}"
        f" · max {max(per_doc.values())}"
    )
    print("\n  by section kind:")
    for kind, n in Counter(c.section_kind for c in chunks).most_common():
        print(f"    {kind:<14} {n:>5}")
    print(f"\n  chunk_id format    <doc_id>:<ordinal>   e.g. {chunks[0].chunk_id}")
    print("    doc_id           sha256(pdf bytes)[:16] — stable across re-ingest")
    print("    ordinal          0-indexed, contiguous within a document")
    if skipped:
        print("\n  skipped:")
        for m in skipped:
            print(f"    {m[:110]}")


if __name__ == "__main__":
    main()
