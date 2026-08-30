"""The paper library: parse once, cache, reuse.

Parsing this corpus takes minutes — tighter word-spacing tolerance is worth its
cost in extraction quality, but not once per evaluation run. Every configuration
in the ablation must see byte-identical documents anyway, so caching is not only
an optimisation: it removes a way for two rows of the table to disagree because
a parser was re-run between them.

Cache entries are keyed by the sha256 of the PDF's bytes, so a re-ingest after
editing a paper's file produces a new entry rather than serving a stale one, and
the same paper added under two filenames is parsed once.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from researchlens.ingest.parse import parse_pdf
from researchlens.types import Document, Section

#: Bumped whenever the parser changes in a way that alters its output. Entries
#: written under an older version are ignored rather than trusted — a cache that
#: survives a parser fix would quietly evaluate the old behaviour.
PARSER_VERSION = 4


def _pdf_key(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _to_json(doc: Document) -> str:
    d = asdict(doc)
    d["_parser_version"] = PARSER_VERSION
    return json.dumps(d)


def _from_json(raw: str) -> Document | None:
    d = json.loads(raw)
    if d.pop("_parser_version", None) != PARSER_VERSION:
        return None
    d["sections"] = [Section(**s) for s in d["sections"]]
    return Document(**d)


def load_library(
    pdf_dir: Path, cache_dir: Path, refresh: bool = False
) -> tuple[list[Document], list[str]]:
    """Parse every PDF in `pdf_dir`, using `cache_dir` where possible.

    Returns the documents and a list of human-readable skip reasons. Papers that
    fail to parse are reported and excluded, never silently indexed: an empty
    document in the index depresses Recall@K with no visible cause.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(p for p in pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDFs in {pdf_dir}")

    docs: list[Document] = []
    skipped: list[str] = []

    for i, path in enumerate(pdfs, start=1):
        entry = cache_dir / f"doc-{_pdf_key(path)}.json"

        if entry.exists() and not refresh:
            doc = _from_json(entry.read_text())
            if doc is not None:
                docs.append(doc)
                print(f"\r  library: {i}/{len(pdfs)} (cached)   ", end="", file=sys.stderr)
                continue

        print(f"\r  library: {i}/{len(pdfs)} parsing {path.name[:40]:<40}", end="", file=sys.stderr)
        try:
            doc = parse_pdf(path)
        except ValueError as e:
            skipped.append(str(e))
            continue
        entry.write_text(_to_json(doc))
        docs.append(doc)

    print(file=sys.stderr)
    return docs, skipped


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Parse a folder of PDFs into the library cache.")
    ap.add_argument("pdf_dir")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true", help="re-parse even if cached")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir).expanduser()
    cache = Path(args.cache) if args.cache else Path(__file__).resolve().parents[2] / "data" / "index"

    docs, skipped = load_library(pdf_dir, cache, refresh=args.refresh)
    for msg in skipped:
        print(f"  skipped: {msg}", file=sys.stderr)

    named = total = 0
    for d in docs:
        for s in d.sections:
            total += 1
            named += s.kind != "other"
    print(
        f"\n{len(docs)} papers · {total} sections "
        f"({named / total * 100:.0f}% named) · {len(skipped)} skipped"
    )


if __name__ == "__main__":
    _main()
