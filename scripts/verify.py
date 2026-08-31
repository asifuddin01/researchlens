"""End-to-end check that the system actually works.

Unit tests cover the parts; this exercises the whole path on the real corpus
and the real models, which is where every defect in this project has actually
come from. Each check prints what it found rather than only pass/fail, because
"retrieval returned 8 results" is true of a broken retriever too.

    python scripts/verify.py              # everything
    python scripts/verify.py --no-live    # skip the network
    python scripts/verify.py --quick      # skip generation (no model needed)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from researchlens.engine import Engine  # noqa: E402
from researchlens.generate.citations import is_grounded  # noqa: E402
from researchlens.live.arxiv import is_live  # noqa: E402

PASS, FAIL, WARN = "  ok  ", " FAIL ", " warn "
_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "", fatal: bool = True) -> bool:
    mark = PASS if ok else (FAIL if fatal else WARN)
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok and fatal:
        _failures.append(name)
    return ok


async def main_async(args) -> int:
    print("\nResearchLens · end-to-end verification\n" + "-" * 62)

    t = time.perf_counter()
    engine = Engine()
    engine.load()
    load_s = time.perf_counter() - t

    check("corpus loads", engine.ready, f"{len(engine.documents)} papers, "
          f"{len(engine.chunks)} passages, {load_s:.1f}s")
    check("no paper indexed empty", all(len(c.text) >= 80 for c in engine.chunks))
    check(
        "every passage can cite itself",
        all(c.doc_title and c.section_heading and c.pages for c in engine.chunks),
    )

    # --- retrieval ------------------------------------------------------
    q = "does deep learning outperform simple linear baselines for perturbation prediction"
    hits, ms = engine.retrieve(q)
    check("retrieval returns results", bool(hits), f"{len(hits)} passages, {ms:.0f} ms")
    check(
        "results are ordered by score",
        [h.score for h in hits] == sorted((h.score for h in hits), reverse=True),
    )
    check(
        "retrieval is on topic",
        any("perturbation" in h.chunk.text.lower() for h in hits),
        hits[0].chunk.doc_title[:52] if hits else "",
    )
    check(
        "both retrievers contribute",
        any("bm25" in h.sources for h in hits) and any("dense" in h.sources for h in hits),
        f"sources: {sorted({s for h in hits for s in h.sources})}",
        fatal=False,
    )
    check("retrieval under 5s", ms < 5000, f"{ms:.0f} ms", fatal=False)

    # --- no bibliography leakage ---------------------------------------
    from researchlens.ingest.chunk import looks_like_bibliography

    leaked = sum(1 for c in engine.chunks if looks_like_bibliography(c.text))
    check("no reference lists indexed", leaked == 0, f"{leaked} found")

    # --- titles ---------------------------------------------------------
    bad_titles = [d.title for d in engine.documents if len(d.title) < 15]
    check("titles extracted", not bad_titles, f"{len(bad_titles)} short: {bad_titles[:3]}")

    if args.quick:
        return _summary()

    # --- generation, grounded ------------------------------------------
    answer = await engine.ask(q, args.provider, live=False)
    check("answer generated", bool(answer.text), f"{answer.generation_ms:.0f} ms")
    check("answer carries citations", bool(answer.citations), f"{len(answer.citations)} cited")
    check("answer is grounded", is_grounded(answer.text, answer.citations))
    check(
        "every citation resolves to a real passage",
        all(c.chunk_id in engine._by_id for c in answer.citations),
    )

    # --- refusal --------------------------------------------------------
    nonsense = "What is the melting point of tungsten carbide in the papers?"
    refusal = await engine.ask(nonsense, args.provider, live=False)
    check(
        "declines what the corpus cannot answer",
        not refusal.citations or "could not find" in refusal.text.lower(),
        f"{len(refusal.citations)} citations",
        fatal=False,
    )

    # --- live search ----------------------------------------------------
    if not args.no_live:
        live_q = "What are the current trends in long-context language models?"
        live = await engine.ask(live_q, args.provider)
        used = [c for c in live.citations if is_live(c.chunk_id)]
        check("live search reaches arXiv", bool(used), f"{len(used)} live citations")
        check(
            "live citations are marked as abstracts",
            all(c.pages == "abstract" for c in used),
            fatal=False,
        )
        check(
            "live evidence is recent",
            all(c.section_heading[-10:] >= "2024-01-01" for c in used),
            used[0].section_heading[-10:] if used else "",
            fatal=False,
        )

    return _summary()


def _summary() -> int:
    print("-" * 62)
    if _failures:
        print(f"\n{len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("\nall checks passed\n")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="local", choices=("local", "hosted"))
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--quick", action="store_true", help="skip generation")
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
