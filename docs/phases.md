# Phases

Referenced from the `NotImplementedError` messages in the source. Each phase
ends at a gate that is a command producing a number, not a judgement that the
work feels done.

## I — The measuring rig

Ground truth before retrieval, so every later component arrives with a measured
delta instead of an intuition.

- [x] `types.py` — record shapes settled first; `Chunk` carries section and page
- [x] `ingest/parse.py` — headings by font geometry, hierarchy, page spans
- [x] `ingest/chunk.py` — passages that can cite themselves
- [x] `ingest/library.py` — parse once, cache, reuse across ablation rows
- [x] `eval/metrics.py` — Recall@K, MRR, nDCG, longhand and unit-tested
- [x] `retrieval/dense.py` — the baseline retriever
- [ ] `eval/questions.jsonl` — **60 hand-written questions.** The long pole, and
      the one file that cannot be generated: if a model writes the questions and
      a model judges the answers, the table measures self-agreement.

**Gate:** `make eval` prints Recall@5, MRR and nDCG@10 for `dense only`.

## II — The retrieval stack

- [ ] `retrieval/bm25.py` — Okapi over a light analyser; no stemming, which
      reliably damages model names like "U-Net" and "scGPT"
- [ ] `retrieval/rerank.py` — cross-encoder over the fused candidates
- [ ] Section-aware retrieval — a question about what a paper *found* should
      prefer Results over Introduction; the metadata is already carried

**Gate:** `make ablation` emits the four-row table into the README.

## III — Grounded answering

- [ ] `generate/ollama.py`, `generate/openai_compat.py`
- [ ] Citation mapping — every marker resolves to a passage, section and page
- [ ] `api/main.py` — the endpoints named in its docstring
- [ ] Faithfulness and citation accuracy added to the harness, reported per
      model over identical retrieved evidence

**Gate:** no claim in an answer lacks a resolvable passage, and citation
accuracy is a measured number.

## IV — Both deployments

- [ ] `docker compose up` on a clean machine, no key, no manual steps
- [ ] Fly deploy: retrieval suspended (<2 GB), generation scaling to zero
- [ ] Cloudflare Worker proxy with a hard daily ceiling
- [ ] `/researchlens` page and a `projects` entry on asifuddin.com

**Gate:** a stranger can clone and run it, or click and use it, without reading
anything first.

---

## Known parser limitations

Recorded because they bound retrieval quality and should be revisited only if
the numbers say they are what caps Recall@5.

**Nature-style descriptive headings.** Papers in Nature and Cell replace IMRaD
headings with sentences — "scGPT improves the precision of cell type
annotation" — and frequently have no explicit "Results" heading to inherit
from. Those passages classify as `other`. They are still retrievable; they just
cite less precisely.

**Page attribution is per section.** A citation says "Results, pp. 7-8" rather
than naming the exact page. Going finer means tracking which page each sentence
landed on through the line grouping, for a citation a reader can already act on.

**Ingest is slow.** Roughly 30 s/paper at `_X_TOLERANCE = 2.0`, dominated by
pdfplumber's character-level extraction. Acceptable because it is cached and
runs once. If it becomes painful, GROBID is both faster and better at this —
see the note in `parse.py`.
