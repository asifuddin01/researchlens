"""Export a portable index: passages and embeddings, without the PDFs.

A deployed container must not carry the corpus. The parser derives each
document id from the PDF's bytes, so `load_library` needs the files present —
which would put roughly 600 MB of journal PDFs into a public image and raise a
redistribution question the project does not need to answer.

Everything serving a query is already downstream of parsing: the passages, the
metadata that lets a citation name a place, and the vectors. That is ~24 MB and
it is what ships.

    python scripts/export_bundle.py                       # everything
    python scripts/export_bundle.py --open-access-only     # for a public demo

**On the public deployment, use `--open-access-only`.** A bundle is extracted
full text. Shipping a private image for your own use is one thing; publishing
one built from subscription journal PDFs is another, and the two are one flag
apart.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from researchlens.ingest import chunk as C  # noqa: E402
from researchlens.ingest.library import PARSER_VERSION, load_library  # noqa: E402
from researchlens.retrieval.dense import DenseRetriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

#: Papers whose full text may be redistributed. arXiv and bioRxiv preprints and
#: PMC open-access articles qualify; a subscription journal PDF does not. The
#: check is on the source filename, which is coarse — hence the warning below.
_OPEN_HINTS = ("arxiv", "biorxiv", "pmc", "plos", "nature communications", "scientific reports")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--open-access-only",
        action="store_true",
        help="include only papers whose filename marks them open access",
    )
    args = ap.parse_args()

    docs, skipped = load_library(ROOT / "data" / "pdfs", ROOT / "data" / "index")
    for m in skipped:
        print(f"  skipped: {m[:100]}", file=sys.stderr)

    if args.open_access_only:
        before = len(docs)
        docs = [
            d for d in docs
            if any(h in Path(d.source_path).name.lower() or h in d.title.lower()
                   for h in _OPEN_HINTS)
        ]
        print(
            f"  open-access filter: {len(docs)} of {before} papers kept.\n"
            "  This matches on filenames and is coarse — check the list before "
            "publishing an image built from it.",
            file=sys.stderr,
        )
        if not docs:
            sys.exit("no papers matched the open-access filter; nothing to export")

    chunks = C.chunk_corpus(docs)
    if not chunks:
        sys.exit("no chunks to export")

    retriever = DenseRetriever()
    retriever.index(chunks)
    matrix = retriever._matrix
    if matrix is None:
        sys.exit("embeddings missing — run a query once to build the cache")

    out = Path(args.out) if args.out else ROOT / "data" / "bundle"
    out.mkdir(parents=True, exist_ok=True)

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "chunking": C.fingerprint(),
                "parser_version": PARSER_VERSION,
                "embedding_model": retriever.model,
                "papers": len(docs),
                "chunks": len(chunks),
                # The order of `chunk_ids` is the row order of the matrix, and
                # nothing else records it. A bundle whose two files disagree
                # would retrieve confidently wrong passages.
                "dim": int(matrix.shape[1]),
            },
            indent=2,
        )
    )

    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id, "doc_id": c.doc_id, "ordinal": c.ordinal,
                        "text": c.text, "section_kind": c.section_kind,
                        "section_heading": c.section_heading,
                        "page_start": c.page_start, "page_end": c.page_end,
                        "doc_title": c.doc_title,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    np.savez_compressed(out / "vectors.npz", matrix=matrix)

    total = sum(f.stat().st_size for f in out.iterdir())
    print(
        f"\n  {len(docs)} papers · {len(chunks)} chunks · {matrix.shape} vectors"
        f"\n  -> {out}  ({total / 1e6:.1f} MB, no PDFs)"
    )


if __name__ == "__main__":
    main()
