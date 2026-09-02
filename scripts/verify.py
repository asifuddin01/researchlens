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

        # Two checks, because the old single one conflated them and then
        # blamed the wrong component. It asserted that the *answer* cited a
        # live passage, and reported "live search reaches arXiv — 0 live
        # citations" on a run where live search had worked perfectly and
        # fetched four passages from three indexes: the model had simply
        # declined to answer from them. Whether live search works is a fact
        # about retrieval; whether the model cites what it is given is a fact
        # about the model, and only the first is this system's to guarantee.
        evidence, _ms = await engine.evidence_for(live_q)
        reached = [r for r in evidence if is_live(r.chunk.chunk_id)]
        sources = sorted({r.chunk.chunk_id.split(":")[0] for r in reached})
        check(
            "live search reaches the prompt",
            bool(reached),
            f"{len(reached)} live passages from {sources or 'nothing'}"
            + (f" — {engine.last_live_error}" if engine.last_live_error else ""),
        )

        live = await engine.ask(live_q, args.provider)
        used = [c for c in live.citations if is_live(c.chunk_id)]
        check(
            "the answer cites live evidence",
            bool(used),
            f"{len(used)} live citations",
            # The model's choice, not the pipeline's. A small model declining
            # to answer from thin evidence is the behaviour this project wants;
            # failing the build for it would train us to want the opposite.
            fatal=False,
        )
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

        # --- the author corpus ------------------------------------------
        #
        # Fetched from the live site, so this belongs with the other network
        # checks. Split the same way live search is: whether the site reaches
        # the prompt is this system's to guarantee, whether the model then
        # writes from it is the model's business.
        from researchlens.live import author as author_source

        about_q = "What research does the author do, and what is he working on?"
        site_chunks = await author_source.fetch()
        check(
            "the author corpus is reachable",
            bool(site_chunks),
            f"{len(site_chunks)} documents from {author_source.CORPUS_URL}"
            + (f" — {author_source.last_error}" if author_source.last_error else ""),
            fatal=False,
        )
        if site_chunks:
            check(
                "the site is not labelled as a paper",
                all(c.pages == "website" for c in site_chunks),
                fatal=False,
            )
            about, _ms = await engine.evidence_for(about_q)
            reached_site = [r for r in about if r.chunk.chunk_id.startswith("site:")]
            check(
                "the site reaches the prompt on a question about the author",
                bool(reached_site),
                f"{len(reached_site)} site passages, top: "
                + (reached_site[0].chunk.doc_title[:40] if reached_site else "none"),
                fatal=False,
            )
            # The failure this guards against is the one that would make every
            # other answer worse: biography leaking into a scientific question.
            technical, _ms = await engine.evidence_for(
                "how does retrieval-augmented generation reduce hallucination?"
            )
            check(
                "the site stays out of a technical question",
                not any(r.chunk.chunk_id.startswith("site:") for r in technical),
            )

        # --- the Elementa ------------------------------------------------
        #
        # The author's own textbook, fetched from the same site. Split the same
        # way as live search: whether it reaches the prompt is this system's to
        # guarantee, whether the model writes from it is the model's business.
        from researchlens.live import elementa as elementa_source

        book = await elementa_source.fetch()
        check(
            "the Elementa is reachable",
            bool(book),
            f"{len(book)} passages from {elementa_source.CORPUS_URL}"
            + (f" — {elementa_source.last_error}" if elementa_source.last_error else ""),
            fatal=False,
        )
        if book:
            check(
                "propositions are not labelled as papers",
                all(c.pages == "proposition" for c in book),
                fatal=False,
            )
            check(
                "every passage of a proposition is separately citable",
                len({c.chunk_id for c in book}) == len(book),
            )
            taught, _ms = await engine.evidence_for(
                "what is a hidden layer actually doing?"
            )
            reached = [r for r in taught if r.chunk.chunk_id.startswith("elementa:")]
            check(
                "the textbook reaches the prompt on a conceptual question",
                bool(reached),
                f"{len(reached)} propositions, top: "
                + (reached[0].chunk.doc_title[:44] if reached else "none"),
                fatal=False,
            )
            # The textbook is about the material, not about the person.
            about_author, _ms = await engine.evidence_for("who is Asif?")
            check(
                "the textbook stays out of a question about the author",
                not any(r.chunk.chunk_id.startswith("elementa:") for r in about_author),
            )

    # --- limitations ----------------------------------------------------
    #
    # Corpus-only, so it runs whether or not live search was asked for. The
    # thing being checked is that a passage where authors concede something
    # reaches the prompt at all: ordinary retrieval ranks it below the method
    # section it follows, because a limitations section is topically further
    # from the question than the work it qualifies.
    lim_q = "what limitations do the authors state for retrieval-augmented generation?"
    lim_ev, _ms = await engine.evidence_for(lim_q)
    conceded = [r for r in lim_ev if Engine._states_a_limitation(r)]
    check(
        "a question about limitations reaches passages that state one",
        bool(conceded),
        f"{len(conceded)} of {len(lim_ev)} passages concede something"
        + (f" — top: {conceded[0].chunk.section_heading[:40]}" if conceded else ""),
    )
    check(
        "a caption is not mistaken for a concession",
        all(r.chunk.section_kind not in Engine._NOT_A_CONCESSION for r in conceded),
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
