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
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
from pathlib import Path

from researchlens.ingest.parse import parse_pdf
from researchlens.types import Document, Section

#: Bumped whenever the parser changes in a way that alters its output. Entries
#: written under an older version are ignored rather than trusted — a cache that
#: survives a parser fix would quietly evaluate the old behaviour.
PARSER_VERSION = 9


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


def _parse_to_cache(args: tuple[str, str, bool]) -> tuple[str, str | None]:
    """Parse one PDF and write its cache entry. Runs in a worker process.

    Workers write the cache themselves and return only a path and an error
    string, so no `Document` is ever pickled back across the process boundary —
    a slotted frozen dataclass is awkward to pickle and there is no reason to
    pay for it when the parent can read the cache.
    """
    path_s, cache_s, refresh = args
    path, cache_dir = Path(path_s), Path(cache_s)
    entry = cache_dir / f"doc-{_pdf_key(path)}.json"
    if entry.exists() and not refresh:
        if _from_json(entry.read_text()) is not None:
            return (path_s, None)
    try:
        doc = parse_pdf(path)
    except ValueError as e:
        return (path_s, str(e))
    entry.write_text(_to_json(doc))
    return (path_s, None)


def load_library(
    pdf_dir: Path, cache_dir: Path, refresh: bool = False, workers: int | None = None
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

    # Parsing is CPU-bound and each paper is independent, so it parallelises
    # cleanly. At 30 papers the serial loop was tolerable; at 100 it is fifty
    # minutes, which is long enough to stop people re-running it — and a
    # measurement people avoid re-running is one that goes stale.
    n_workers = workers if workers is not None else min(os.cpu_count() or 4, 8)
    jobs = [(str(p), str(cache_dir), refresh) for p in pdfs]

    def serial() -> None:
        for i, job in enumerate(jobs, start=1):
            _path_s, err = _parse_to_cache(job)
            if err:
                skipped.append(err)
            print(f"\r  library: {i}/{len(pdfs)}   ", end="", file=sys.stderr)

    if n_workers > 1 and len(pdfs) > 1:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                for i, (_path_s, err) in enumerate(
                    pool.map(_parse_to_cache, jobs, chunksize=1), start=1
                ):
                    if err:
                        skipped.append(err)
                    print(f"\r  library: {i}/{len(pdfs)}   ", end="", file=sys.stderr)
        except (BrokenProcessPool, ImportError, RuntimeError) as e:
            # macOS spawns workers by re-importing __main__, which fails when
            # there is no importable entry point — a REPL, a notebook, a
            # heredoc. Falling back keeps `load_library` callable from
            # anywhere; the only cost is that the first parse is slower.
            print(
                f"\r  library: parallel parsing unavailable ({type(e).__name__}); "
                "falling back to one process",
                file=sys.stderr,
            )
            skipped.clear()
            serial()
    else:
        serial()

    print(file=sys.stderr)

    # Read back in a stable order, so the chunk ordinals every hand-written
    # label refers to do not depend on which worker finished first.
    for path in pdfs:
        entry = cache_dir / f"doc-{_pdf_key(path)}.json"
        if not entry.exists():
            continue
        doc = _from_json(entry.read_text())
        if doc is not None:
            docs.append(doc)

    return docs, skipped


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Parse a folder of PDFs into the library cache.")
    ap.add_argument("pdf_dir")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true", help="re-parse even if cached")
    ap.add_argument("--workers", type=int, default=None, help="parser processes (default: cores)")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir).expanduser()
    cache = Path(args.cache) if args.cache else Path(__file__).resolve().parents[2] / "data" / "index"

    docs, skipped = load_library(pdf_dir, cache, refresh=args.refresh, workers=args.workers)
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
