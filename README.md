# ResearchLens

Evidence-grounded retrieval over scientific literature. Every claim in an answer
resolves to a passage — a paper, a section, a page — and every architectural
choice has to justify itself with a number.

Runs entirely on your machine. No API key, no document leaving the host.

```bash
git clone https://github.com/asifuddin01/researchlens
cd researchlens
docker compose up
```

Then open <http://localhost:8000>. The first run downloads a ~2 GB model once.

---

## Why this is not a RAG demo

Three things, in the order they were built.

**The measuring rig came first.** Sixty questions over an open-access corpus,
each hand-labelled with the passages that genuinely answer it. Written before
any retrieval code, against a deliberately weak baseline, so that every
component added afterwards arrives with a measured delta rather than an
intuition.

**The ground truth is written by hand.** If a language model writes the
questions and a language model judges the answers, the table below measures a
model's agreement with itself. `eval/questions.jsonl` is the most valuable file
in this repository and the most tedious one to produce.

**The table is generated.** `make ablation` runs every configuration through
one code path and rewrites this file. Nothing here is typed by a human, so
nothing here can drift away from the code.

## Retrieval ablation

Each row adds exactly one component to the row above it, over an identical
parser, chunker and index — so the difference between adjacent rows is
attributable to the named component and to nothing else.

<!-- ablation:start -->

_Not yet measured. Run `make corpus && make ablation`._

<!-- ablation:end -->

Latency is in the table on purpose. A component that buys three points of
recall for 8 ms and one that buys three points for 400 ms are different
engineering decisions, and a table reporting only quality hides which is on
offer.

## How it works

```
PDF ─ parse ─ chunk ─┬─ BM25   ─┐
   (sections,        │          ├─ RRF ─ cross-encoder ─ context ─ LLM ─ answer
    page spans)      └─ dense  ─┘                                        + citations
```

**Structure first.** A PDF is not a string. `researchlens/ingest/parse.py` finds
headings by font geometry — papers set headings larger or bolder than body text,
a signal present in essentially every typeset paper — so a passage knows which
section it came from and which page it was printed on. That is what makes a
citation nameable rather than approximate.

**Hybrid, because the two halves fail differently.** A dense encoder maps
"Dice 0.91", "SwinUNETR" and "KiTS23" into neighbourhoods of things that mean
something similar, which for a metric value or a dataset name is exactly wrong.
BM25 matches them literally and is helpless at paraphrase. Reciprocal rank
fusion combines them by position rather than score, so no normalisation has to
be tuned per corpus.

**Reranking, provisionally.** A bi-encoder never sees query and passage
together; a cross-encoder scores the pair jointly and can notice the right
metric on the wrong dataset. It may not help here — on a small corpus where
hybrid retrieval already puts the labelled passage in the top five there is
little room above it. If the ablation says so, that row stays in the table
unchanged.

## Local and hosted

The default generator is local (Ollama). A hosted OpenAI-compatible endpoint is
optional and stays optional — with no key configured, the local model serves
every request.

Because retrieval is identical on both sides, switching provider isolates
generation. That makes the toggle a live ablation rather than a convenience,
and the harness reports faithfulness and citation accuracy per model over the
same retrieved evidence.

## The public demo

<https://asifuddin.com/researchlens> is the project's page; the system itself
runs at [asifuddin01/researchlens](https://huggingface.co/spaces/asifuddin01/researchlens)
on a Hugging Face Space. Same engine, same grounding rules, same 101 papers —
generation runs on the Space's attached GPU rather than a local model.

It cannot honestly claim "your documents never leave your machine", so it does
not. What it does claim, and what the code enforces, is narrower and true: a
paper you add there is parsed into an index belonging to **your browser session
alone**, held in memory, never written to disk, never merged into the public
corpus, and dropped when the session goes idle.

That is not caution, it is correctness. A Space is *one process serving every
visitor*, so appending an upload to the shared index would put a stranger's
manuscript in someone else's answers, with a citation, indistinguishable from a
paper that belongs there. `researchlens/uploads.py` exists to make that
impossible rather than unlikely.

### Adding papers

```
POST /ingest        multipart: file=@paper.pdf, session=<optional>
                    → {"session": "...", "doc_id": "...", "title": "...", ...}
POST /ask           {"question": "...", "session": "...", "doc_ids": [...]}
DELETE /ingest/{session}
```

The session id is returned by the first upload and passed back on later calls.
It is the only thing separating one reader's papers from another's, so it is
generated server-side and checked for shape, never accepted as a free string.

Limits, and why: 20 MB and 80 pages per paper, 5 papers per session, one hour
idle. Parsing and embedding cost real CPU on shared hardware — a sixteen-page
paper is about seven seconds on a laptop — and bounded is not the same as free.

## Layout

```
researchlens/
  types.py            record shapes; everything depends on these and they on nothing
  config.py           every value has a working default
  ingest/parse.py     PDF → sections with page spans
  ingest/chunk.py     sections → passages that can cite themselves
  retrieval/          bm25 · dense · fusion · rerank
  retrieval/pipeline.py   the ablation axis — one code path, four configurations
  uploads.py          per-session papers, never merged into the corpus
  generate/           provider protocol + local and hosted implementations
  api/                FastAPI; the local UI and the demo share it
eval/
  corpus.yaml         the papers, open-access only
  questions.jsonl     hand-written ground truth
  metrics.py          Recall@K · MRR · nDCG, longhand and unit-tested
  run.py              the harness
```

## Development

```bash
make install                     # venv + dependencies
make corpus                      # fetch the open-access papers
make ingest DIR=~/my/papers      # or parse your own folder (cached)
make search Q="how is a GRN inferred"   # inspect retrieval directly
make test                        # unit tests
make eval                        # score the baseline
make ablation                    # regenerate the table above
```

`make search` is worth using before trusting any number: reading what actually
comes back for a query separates a chunking problem from a retrieval problem,
and it needs no ground truth to be informative.

Python 3.11 or newer, 3.14 included. The container pins 3.12 — a reproducible
deployment wants a known-good interpreter rather than the newest one — but
nothing here requires it.

## Licence

MIT.
