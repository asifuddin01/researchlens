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
- [x] `retrieval/dense.py` — the baseline retriever, embeddings cached
- [x] `scripts/search.py` — inspect retrieval without ground truth
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

## V — Beyond the corpus (added after Benchmark B)

Benchmark B asks what a field is doing *now*. A fixed corpus cannot answer
that however good its retrieval, and the failure is the dangerous kind: asked
for current trends in large language models, the system answered from Attention
Is All You Need (2017) and BERT (2018) with real citations and no sign the
evidence was eight years old.

- [x] `live/arxiv.py` — search by submission date, not relevance; arXiv's
      relevance ordering happily returns the seminal 2017 paper first, which is
      the exact failure being fixed
- [x] Live results adapted into `Chunk`, so fusion, prompting and citation
      resolution work unchanged, with the honesty carried in the fields:
      `pages` reads "abstract", the heading names the arXiv id and date
- [x] The prompt marks live passages ABSTRACT ONLY — an abstract supports what
      a paper *claims*, not what it measured, on which dataset, with what result
- [x] The decision to go live is made from measured corpus support, counted in
      distinct papers rather than passages: eight passages from one paper is
      one paper's worth of evidence
- [ ] Semantic Scholar, for the journal-only literature arXiv misses — most
      clinical and much biological work
- [ ] Cache live results, so a repeated question does not re-fetch

**Gate:** a question with no corpus support returns recent, dated, correctly
labelled evidence — or declines, and never answers from model memory.

## IV — Both deployments

- [ ] `docker compose up` on a clean machine, no key, no manual steps
- [ ] Fly deploy: retrieval suspended (<2 GB), generation scaling to zero
- [ ] Cloudflare Worker proxy with a hard daily ceiling
- [x] `/researchlens` page and a `projects` entry on asifuddin.com — the page is
      live and reads its endpoint from `PUBLIC_RESEARCHLENS_API`, so deploying
      the backend turns the question box on with no code change

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

**Embedding throughput.** Measured on 192 real passages (bge-small, 8 cores):

| configuration | rate | full corpus |
|---|---|---|
| fastembed default | 2.2/s | 25 min |
| `threads=8` | 4.0/s | 14 min |
| **CoreML provider** | **5.7/s** | **9.8 min** |
| `threads=8, parallel=0` | 1.2/s | 48 min |

Two results worth keeping. CoreML is the clear winner where it exists, so it is
requested by name and detected rather than assumed — the container runs Linux,
where it does not exist. And fastembed's `parallel` multiprocessing is *slower*:
each worker loads its own copy of the model, and at this corpus size that reload
dominates. It is deliberately unused.

Even at 5.7/s this is not fast for a 33M-parameter model. It is tolerable only
because embeddings are cached on a key covering model, instruction, chunk ids
and text. If the corpus grows much past 100 papers, this is the second thing
that needs attention after parsing.

**Ingest is slow.** Roughly 30 s/paper at `_X_TOLERANCE = 2.0`, dominated by
pdfplumber's character-level extraction; 30 papers take ~15 minutes. Embedding
those 3,364 passages with bge-small on CPU is a further several minutes. Both
are cached and run once, which is why neither has been optimised — but if the
corpus grows past ~100 papers, ingest is the first thing that will need to
become parallel, and GROBID is both faster and better at the parsing half.

**Corpus composition.** 30 papers, 1,021 sections, 3,364 passages, 61% of them
in a named section. Books were excluded deliberately: five of them produced
~5,800 further passages and would dominate the index, and a 487-page monograph
is a different retrieval problem from a 12-page paper.
