"""Turn the two written benchmarks into files the harness can read.

Benchmark A asks about papers in the corpus. Its questions arrive as templates
— "What dataset(s) are used in the study?" names no study — so each is bound to
a specific paper, or to a pair, or to the corpus, depending on what it asks.

`relevant_chunks` is left empty on purpose and is *not* filled by this script.
Retrieval metrics need passages a person judged relevant by reading. Generating
them by running retrieval and keeping the top hits would label the system
correct by construction, and every configuration in the ablation would then
score well. That is the one shortcut this project cannot take.

Benchmark B asks about the literature at large — current trends, open problems,
what a field is converging on. A fixed 101-paper corpus cannot answer most of
it, and that is what makes it useful: the correct behaviour is to answer within
the evidence or to decline, never to produce a fluent survey from the model's
own memory. So each is classified by what the corpus can actually support, and
that classification is scoreable today, with no hand-labelling at all.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = Path.home() / "Downloads"

# Anchor papers, chosen to span both halves of the corpus so a bound question
# is never answerable from only one field.
ANCHORS = [
    ("93b9a09b1d7d2c9e", "Deep-learning-based gene perturbation effect prediction"),
    ("bdfaa68d8984f0dc", "Attention Is All You Need"),
    ("5692a5514787a8c6", "BERT"),
    ("c31585561f771c57", "scGPT"),
    ("23e3249e9a1e7541", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"),
    ("eaeaeef08d3b63b2", "Benchmarking algorithms for generalizable single-cell perturbation"),
    ("7d9f878c23b460e4", "Chain-of-Thought Prompting"),
    ("e736f785739c6b70", "A Privacy-Preserving Collaborative Federated Learning Framework"),
]

# Benchmark A, by question number, into what the question needs.
#   single     one paper
#   pair       two papers
#   corpus     the whole corpus
#   verify     a claim to check — the claim itself must be supplied by hand
A_SHAPE = [
    (1, 30, "single"),
    (31, 40, "pair"),
    (41, 50, "corpus"),
    (51, 80, "verify"),
]

# Benchmark B: what the corpus can actually support, measured rather than
# guessed. The first version of this was a hand-written list of question
# numbers, and it was wrong in a whole cluster — questions about LLM trends,
# agents and vision were marked "beyond corpus" when the corpus holds 30
# language-model papers. The system answered them correctly, with real
# citations, and scored 20% against my classification rather than its own
# behaviour.
#
# So coverage is now counted from the corpus's own paper titles. This is a
# *proposal*, not ground truth: it uses a lexical signal, and the retriever
# being evaluated also uses one, so accepting it unreviewed would leak the
# answer into the question. Every row carries the papers it matched, so the
# classification can be checked by reading rather than trusted.
TOPICS: dict[str, list[str]] = {
    "language models": ["language model", "llm", "gpt", "bert", "transformer",
                        "mixtral", "instruction", "attention is all"],
    "reasoning": ["chain-of-thought", "chain of thought", "reasoning", "verify step"],
    "efficient training": ["efficient", "lora", "quantiz", "distill", "sparse",
                           "mixture of experts", "flash"],
    "retrieval-augmented generation": ["retrieval-augmented", "retrieval augmented"],
    "long context": ["long context", "long-context", "context window"],
    "agents": ["agent", "autogen", "tool use"],
    "multimodal": ["multimodal", "vision-language", "vision language", "clip",
                   "blip", "flamingo"],
    "computer vision": ["vision transformer", "image recognition",
                        "masked autoencoder", "swin", "segment anything", "16x16"],
    "3d vision": ["3d ", "point cloud", "nerf"],
    "medical imaging": ["medical", "radiology", "retinopathy", "clinical", "diabetic"],
    "single-cell": ["single-cell", "single cell", "scgpt", "geneformer",
                    "transcriptom", "chromatin"],
    "perturbation prediction": ["perturbation", "crispr", "gears", "cpa", "in silico"],
    "gene regulatory networks": ["regulatory network", "grn"],
    "causal inference": ["causal", "causation", "intervention", "dag", "counterfactual"],
    "federated learning": ["federated", "privacy-preserving", "homomorphic"],
    "pathology": ["patholog", "histolog", "whole slide"],
    "renal imaging": ["renal", "kidney"],
    "report generation": ["report generation", "captioning"],
    "hallucination": ["hallucinat", "factual", "faithful"],
    "evaluation": ["benchmark", "evaluat", "metric"],
}

#: Papers on a topic, below which the corpus cannot support a survey of it.
#: Four is a judgement: one paper is an anecdote, and a "current trends"
#: question answered from three papers is a reading list, not a trend.
_SURVEY_FLOOR = 4


def topics_for(question: str) -> list[str]:
    q = question.lower()
    return [name for name, keys in TOPICS.items() if any(k in q for k in keys)]


def corpus_topic_counts(titles: list[str]) -> dict[str, int]:
    lowered = [t.lower() for t in titles]
    return {
        name: sum(1 for t in lowered if any(k in t for k in keys))
        for name, keys in TOPICS.items()
    }


def _parse(path: Path) -> list[tuple[int, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*(\d{1,3})\.\s+(.*\S)\s*$", line)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


def _shape(n: int) -> str:
    for lo, hi, kind in A_SHAPE:
        if lo <= n <= hi:
            return kind
    return "single"


def build_a(rows: list[tuple[int, str]]) -> list[dict]:
    out = []
    for n, text in rows:
        shape = _shape(n)
        rec: dict = {
            "id": f"A{n:03d}",
            "benchmark": "A",
            "shape": shape,
            "template": text,
            "question": text,
            "relevant_chunks": [],
            "needs_labels": True,
        }
        if shape == "single":
            doc_id, title = ANCHORS[(n - 1) % len(ANCHORS)]
            rec["doc_ids"] = [doc_id]
            rec["question"] = text.replace("the paper", f"“{title}”").replace(
                "the study", f"“{title}”"
            )
        elif shape == "pair":
            a = ANCHORS[(n - 31) % len(ANCHORS)]
            b = ANCHORS[(n - 31 + 4) % len(ANCHORS)]
            rec["doc_ids"] = [a[0], b[0]]
            rec["question"] = (
                text.replace("two papers", f"“{a[1]}” and “{b[1]}”")
                .replace("the two papers", f"“{a[1]}” and “{b[1]}”")
            )
        elif shape == "corpus":
            rec["doc_ids"] = []
        else:  # verify
            rec["doc_ids"] = []
            rec["needs_claim"] = True
            rec["claim"] = ""
        out.append(rec)
    return out


def build_b(rows: list[tuple[int, str]], counts: dict[str, int]) -> list[dict]:
    out = []
    for n, text in rows:
        matched = topics_for(text)
        support = {t: counts.get(t, 0) for t in matched}
        best = max(support.values(), default=0)

        if best == 0:
            coverage, expect = "beyond_corpus", "refuse"
        elif best < _SURVEY_FLOOR:
            # Something is there, but too little to survey from. The correct
            # output says what the few papers show and declines the rest.
            coverage, expect = "thin", "scoped"
        else:
            coverage, expect = "partial", "scoped"

        out.append(
            {
                "id": f"B{n:03d}",
                "benchmark": "B",
                "question": text,
                "topics": matched,
                "corpus_support": support,
                "coverage": coverage,
                # answer — grounded answer with citations
                # scoped — answers what the corpus supports and says so
                # refuse — declines rather than writing from model memory
                "expect": expect,
                "reviewed": False,
                "relevant_chunks": [],
                "needs_labels": coverage != "beyond_corpus",
            }
        )
    return out


def write(path: Path, meta: dict, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {len(rows):>3} questions -> {path.name}")


def _corpus_titles() -> list[str]:
    """Paper titles from the chunk dump, or from the library if it is absent."""
    dump = ROOT / "chunks-c1000-o180.jsonl"
    if dump.exists():
        seen = {}
        for line in dump.open(encoding="utf-8"):
            o = json.loads(line)
            if not o.get("_meta"):
                seen[o["doc_id"]] = o["doc_title"]
        return list(seen.values())

    from researchlens.ingest.library import load_library

    docs, _ = load_library(ROOT / "data" / "pdfs", ROOT / "data" / "index")
    return [d.title for d in docs]


def main() -> None:
    a_src = DOWNLOADS / "Benchmark_A_80_Questions (1).txt"
    b_src = DOWNLOADS / "Benchmark_B_Open_World_Research_Questions.txt"
    for p in (a_src, b_src):
        if not p.exists():
            sys.exit(f"missing {p}")

    titles = _corpus_titles()
    counts = corpus_topic_counts(titles)
    a = build_a(_parse(a_src))
    b = build_b(_parse(b_src), counts)

    write(
        ROOT / "eval" / "benchmark_a.jsonl",
        {
            "_meta": True,
            "benchmark": "A",
            "about": "questions about papers in the corpus",
            "chunking": "c1000-o180",
            "labelled_by": "hand",
            "note": "relevant_chunks are written by reading, never by running retrieval",
        },
        a,
    )
    write(
        ROOT / "eval" / "benchmark_b.jsonl",
        {
            "_meta": True,
            "benchmark": "B",
            "about": "open-world questions; measures answering within evidence vs declining",
            "chunking": "c1000-o180",
        },
        b,
    )

    from collections import Counter

    print("\n  A by shape:   ", dict(Counter(r["shape"] for r in a)))
    print("  B by coverage:", dict(Counter(r["coverage"] for r in b)))
    print("\n  Topic support in the corpus (papers per topic):")
    for t, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        flag = "" if c >= _SURVEY_FLOOR else ("  <- nothing" if c == 0 else "  <- thin")
        print(f"    {t:<32} {c:>3}{flag}")
    print(f"\n  scoreable today: {sum(1 for r in b if r['expect'] == 'refuse')} refusal questions")
    print(f"  awaiting labels: {sum(1 for r in a if r['needs_labels'])} in A"
          f" + {sum(1 for r in b if r['needs_labels'])} in B")


if __name__ == "__main__":
    main()
