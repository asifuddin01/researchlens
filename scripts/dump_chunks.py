"""Write every chunk to a file, for hand-labelling.

    python scripts/dump_chunks.py                 # JSONL, full text
    python scripts/dump_chunks.py --format txt    # readable, for skimming

JSONL rather than CSV because chunk text contains commas, quotes and newlines,
and a CSV of scientific prose is a quoting bug waiting to happen. One JSON
object per line stays greppable — `grep '"linear baseline"' chunks.jsonl` works
— while surviving any punctuation a paper contains.

The header line records the chunking fingerprint, so a dump can always be
matched back to the parameters that produced its ids.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from researchlens.ingest import chunk as C  # noqa: E402
from researchlens.ingest.library import PARSER_VERSION, load_library  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", choices=("jsonl", "txt"), default="jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    docs, skipped = load_library(ROOT / "data" / "pdfs", ROOT / "data" / "index")
    chunks = C.chunk_corpus(docs)

    out = Path(args.out) if args.out else ROOT / f"chunks-{C.fingerprint()}.{args.format}"

    with out.open("w", encoding="utf-8") as f:
        if args.format == "jsonl":
            f.write(
                json.dumps(
                    {
                        "_meta": True,
                        "chunking": C.fingerprint(),
                        "parser_version": PARSER_VERSION,
                        "papers": len(docs),
                        "chunks": len(chunks),
                        "skipped": len(skipped),
                    }
                )
                + "\n"
            )
            for c in chunks:
                f.write(
                    json.dumps(
                        {
                            "chunk_id": c.chunk_id,
                            "doc_id": c.doc_id,
                            "ordinal": c.ordinal,
                            "doc_title": c.doc_title,
                            "section_kind": c.section_kind,
                            "section_heading": c.section_heading,
                            "pages": c.pages,
                            "chars": len(c.text),
                            "text": c.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        else:
            f.write(
                f"ResearchLens chunks · {C.fingerprint()} · parser v{PARSER_VERSION}\n"
                f"{len(docs)} papers · {len(chunks)} chunks\n"
                + "=" * 78
                + "\n\n"
            )
            current = None
            for c in chunks:
                if c.doc_id != current:
                    current = c.doc_id
                    f.write(f"\n{'=' * 78}\n{c.doc_title}\n  doc_id {c.doc_id}\n{'=' * 78}\n\n")
                f.write(
                    f"[{c.chunk_id}]  {c.section_kind} · {c.section_heading} · p{c.pages}\n"
                )
                f.write(" ".join(c.text.split()) + "\n\n")

    size = out.stat().st_size
    print(f"{len(chunks)} chunks -> {out}  ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
