"""Produce a review sheet for Benchmark B's coverage labels.

Whether a question is answerable from the corpus is the one judgement in this
benchmark, and it decides the score entirely. Two automated attempts failed the
same way — a hand-written list of question numbers, then keyword matching
against paper titles — and both marked questions beyond-corpus that the system
then answered correctly from papers that are plainly indexed. "When do deep
learning methods fail to outperform simpler baselines?" was labelled
unanswerable while the paper titled *Deep-learning-based gene perturbation
effect prediction does not yet outperform simple linear baselines* sat in the
corpus.

So the label is set by reading, and this makes the reading fast: each question
appears with what the system actually answered and which papers it cited, which
is most of what deciding requires.

    python scripts/review_benchmark_b.py > review.md

Edit `expect` in eval/benchmark_b.jsonl to one of:
    refuse   nothing in the corpus supports this
    scoped   partly covered; answering the covered part and saying so is right
    answer   the corpus covers this and a grounded answer is expected
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "benchmark_b.jsonl"
RESULTS = ROOT / "eval" / "results_benchmark_b.jsonl"


def load(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"missing {path}")
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not obj.get("_meta"):
            out.append(obj)
    return out


def main() -> None:
    questions = {q["id"]: q for q in load(QUESTIONS)}
    results = {r["qid"]: r for r in load(RESULTS)}

    print("# Benchmark B — coverage review\n")
    print("Each question with its current label and what the system actually did.")
    print("Change `expect` in `eval/benchmark_b.jsonl` where the label is wrong.\n")
    print("Disagreements first — these are where the label and the behaviour differ,")
    print("and where a wrong label is doing the most damage to the score.\n")

    rows = []
    for qid, q in questions.items():
        r = results.get(qid)
        rows.append((not (r and r.get("correct", True)), qid, q, r))
    rows.sort(key=lambda t: (not t[0], t[1]))

    for disagreed, qid, q, r in rows:
        mark = "**DISAGREES**" if disagreed else "agrees"
        print(f"\n---\n\n### {qid} — {mark}\n")
        print(f"**Q:** {q['question']}\n")
        print(f"- label: `expect: {q['expect']}` (coverage: {q['coverage']})")
        if q.get("corpus_support"):
            support = ", ".join(f"{k}={v}" for k, v in q["corpus_support"].items())
            print(f"- topics matched: {support}")
        else:
            print("- topics matched: none")
        if r:
            print(f"- system: **{r['got']}**, {r['citations']} citations, {r['seconds']:.0f}s")
            answer = " ".join(r["text"].split())
            print(f"\n> {answer[:340]}{'…' if len(answer) > 340 else ''}")
        else:
            print("- system: not run")


if __name__ == "__main__":
    main()
