"""Score whether the system answers within its evidence or declines.

This is the half of Benchmark B that needs no hand-labelling. For a question
the corpus cannot support, the correct output is a refusal, and checking that
requires only the question's expected behaviour — not a list of relevant
passages.

It is also the property that decides whether the tool is usable at all. A
system that answers "what are the current trends in long-context language
models?" from a 101-paper corpus is not retrieving; it is reciting what the
model already believed, and a reader cannot tell the two apart. Retrieval
quality is worth nothing without this.

    python -m eval.refusal --provider local
    python -m eval.refusal --provider local --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from researchlens.engine import Engine

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "benchmark_b.jsonl"


@dataclass(frozen=True, slots=True)
class Outcome:
    qid: str
    question: str
    expect: str
    #: "answer" when the system produced a cited answer, "refuse" otherwise.
    got: str
    correct: bool
    citations: int
    text: str
    seconds: float


def load(path: Path = QUESTIONS) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("_meta"):
            continue
        rows.append(obj)
    if not rows:
        sys.exit(f"no questions in {path} — run scripts/build_benchmarks.py first")
    return rows


async def run(engine: Engine, rows: list[dict], provider: str) -> list[Outcome]:
    out: list[Outcome] = []
    for i, q in enumerate(rows, start=1):
        started = time.perf_counter()
        answer = await engine.ask(q["question"], provider)
        elapsed = time.perf_counter() - started

        got = "answer" if answer.citations else "refuse"
        # A "scoped" expectation is satisfied either way: answering the part
        # the corpus covers is right, and declining is defensible. Only the
        # unambiguous ends are scored, so the number means something.
        if q["expect"] == "scoped":
            correct = True
        else:
            correct = got == q["expect"]

        out.append(
            Outcome(
                qid=q["id"], question=q["question"], expect=q["expect"], got=got,
                correct=correct, citations=len(answer.citations),
                text=answer.text, seconds=elapsed,
            )
        )
        mark = "ok " if correct else "MISS"
        print(
            f"  [{i:>3}/{len(rows)}] {mark} {q['id']} expect={q['expect']:<7}"
            f" got={got:<7} cites={len(answer.citations):<2} {elapsed:5.1f}s",
            file=sys.stderr,
            flush=True,
        )
    return out


def report(outcomes: list[Outcome]) -> None:
    def group(expect: str) -> list[Outcome]:
        return [o for o in outcomes if o.expect == expect]

    refuse, answer = group("refuse"), group("answer")
    print("\n" + "=" * 66)
    print("  Benchmark B — answering within evidence")
    print("=" * 66)

    for label, rows in (("beyond corpus (should refuse)", refuse),
                        ("in corpus (should answer)", answer)):
        if not rows:
            continue
        ok = sum(1 for o in rows if o.correct)
        print(f"\n  {label:<34} {ok}/{len(rows)}  ({ok / len(rows) * 100:.0f}%)")

    scored = refuse + answer
    if scored:
        ok = sum(1 for o in scored if o.correct)
        print(f"\n  {'overall':<34} {ok}/{len(scored)}  ({ok / len(scored) * 100:.0f}%)")

    med = sorted(o.seconds for o in outcomes)[len(outcomes) // 2]
    print(f"  {'median latency':<34} {med:.1f}s")

    misses = [o for o in scored if not o.correct]
    if misses:
        print(f"\n  misses ({len(misses)}):")
        for o in misses[:12]:
            print(f"    {o.qid} expect={o.expect} got={o.got} · {o.question[:58]}")
            print(f"       {' '.join(o.text.split())[:100]}")


async def main_async(args) -> None:
    engine = Engine()
    print("  loading corpus…", file=sys.stderr)
    engine.load()
    print(
        f"  {len(engine.documents)} papers · {len(engine.chunks)} passages\n",
        file=sys.stderr,
    )

    rows = load()
    if args.only:
        rows = [r for r in rows if r["expect"] == args.only]
    if args.limit:
        rows = rows[: args.limit]

    outcomes = await run(engine, rows, args.provider)
    report(outcomes)

    if args.out:
        Path(args.out).write_text(
            "\n".join(json.dumps(o.__dict__ if hasattr(o, "__dict__") else {
                "qid": o.qid, "question": o.question, "expect": o.expect,
                "got": o.got, "correct": o.correct, "citations": o.citations,
                "text": o.text, "seconds": round(o.seconds, 2),
            }) for o in outcomes)
        )
        print(f"\n  written to {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="local", choices=("local", "hosted"))
    ap.add_argument("--only", choices=("refuse", "answer", "scoped"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
