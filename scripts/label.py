"""Browse the corpus by reading, to write ground truth by hand.

    python scripts/label.py --papers                 # every paper, with its id
    python scripts/label.py --paper geneformer       # that paper's passages
    python scripts/label.py --grep "linear baseline" # passages containing a phrase
    python scripts/label.py --chunk 3f2a9c1b04e7:12  # one passage in full

Deliberately *not* a retriever. Nothing here ranks by embedding similarity,
because the labels this produces are what the embedding retriever is scored
against — using it to find them would make the system correct by construction
and every ablation row would look good.

The safe order is: write the question from what you know the paper says, then
find the passage, then record its id. The unsafe order is running a search and
writing a question that fits whatever came back. `--grep` is a lookup, not a
ranking, and is fine for the first order and dangerous for the second.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from researchlens.ingest.chunk import chunk_corpus  # noqa: E402
from researchlens.ingest.library import load_library  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _load():
    docs, skipped = load_library(ROOT / "data" / "pdfs", ROOT / "data" / "index")
    for m in skipped:
        print(f"  skipped: {m}", file=sys.stderr)
    return docs, chunk_corpus(docs)


def _show(c, chars: int) -> None:
    text = " ".join(c.text.split())
    print(f"  {c.chunk_id:<20} {c.section_kind:<12} p{c.pages:<7} {c.section_heading[:38]}")
    print(f"    {text[:chars]}{'...' if len(text) > chars else ''}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--papers", action="store_true", help="list papers and their ids")
    ap.add_argument("--paper", help="show one paper's passages (id or title fragment)")
    ap.add_argument("--grep", help="passages containing this text (case-insensitive)")
    ap.add_argument("--chunk", help="print one passage in full")
    ap.add_argument("--section", help="restrict to a section kind, e.g. results")
    ap.add_argument("--chars", type=int, default=220)
    args = ap.parse_args()

    docs, chunks = _load()

    if args.papers:
        print(f"\n{len(docs)} papers\n")
        for d in sorted(docs, key=lambda d: d.title):
            n = sum(1 for c in chunks if c.doc_id == d.doc_id)
            print(f"  {d.doc_id}  {n:>4} passages  {d.title[:66]}")
        return

    if args.chunk:
        for c in chunks:
            if c.chunk_id == args.chunk:
                print(f"\n{c.doc_title}\n{c.section_heading} · p{c.pages} · {c.section_kind}\n")
                print(" ".join(c.text.split()))
                return
        sys.exit(f"no passage with id {args.chunk}")

    sel = chunks
    if args.paper:
        q = args.paper.lower()
        sel = [c for c in sel if q in c.doc_id.lower() or q in c.doc_title.lower()]
        if not sel:
            sys.exit(f"no paper matching {args.paper!r} — try --papers")
    if args.section:
        sel = [c for c in sel if c.section_kind == args.section]
    if args.grep:
        pat = re.compile(re.escape(args.grep), re.I)
        sel = [c for c in sel if pat.search(c.text)]

    print(f"\n{len(sel)} passages\n")
    for c in sel:
        _show(c, args.chars)


if __name__ == "__main__":
    main()
