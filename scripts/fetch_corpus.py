"""Download the evaluation corpus.

The PDFs are not committed. They are large, and redistributing them raises a
licensing question this repository does not need to answer — every paper in
`eval/corpus.yaml` is open-access at its source, so fetching is both cheap and
unambiguous.

Deliberately polite: arXiv asks automated clients to space requests, and a
corpus of twenty-five papers is not worth being rate-limited over.
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus.yaml"
OUT = ROOT / "data" / "pdfs"

#: arXiv's robots guidance asks for a gap between requests from a script.
DELAY_SECONDS = 3


def main() -> None:
    spec = yaml.safe_load(CORPUS.read_text())
    papers = spec.get("papers", [])
    OUT.mkdir(parents=True, exist_ok=True)

    fetched = skipped = failed = 0
    for paper in papers:
        dest = OUT / f"{paper['id']}.pdf"
        if dest.exists():
            print(f"  have  {paper['id']}")
            skipped += 1
            continue

        url = f"https://arxiv.org/pdf/{paper['arxiv']}"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ResearchLens/0.1 (evaluation corpus; contact via GitHub)"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("response was not a PDF")
            dest.write_bytes(data)
            print(f"  got   {paper['id']}  ({len(data) / 1e6:.1f} MB)")
            fetched += 1
        except Exception as e:  # noqa: BLE001 — report and continue
            print(f"  FAIL  {paper['id']}: {e}", file=sys.stderr)
            failed += 1
        time.sleep(DELAY_SECONDS)

    print(f"\n{fetched} fetched, {skipped} already present, {failed} failed → {OUT}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
