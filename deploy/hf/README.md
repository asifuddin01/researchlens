---
title: ResearchLens
emoji: 🔬
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: mit
short_description: Evidence-grounded retrieval over 101 papers, with citations
---

# ResearchLens

Evidence-grounded retrieval over scientific literature. Every claim in an
answer resolves to a passage — a paper, a section, a page — and every part of
the retrieval stack had to justify itself with a measured number.

Ask across all 101 papers, confine the answer to the ones you choose, or add
your own PDFs and ask about those.

## Your papers

A paper you upload is parsed, chunked and embedded into an index belonging to
**your browser session alone**. It is held in memory, never written to disk,
never merged into the public corpus, and dropped when the session goes idle.

That is a correctness requirement, not a courtesy. A Space is one process
serving every visitor, so appending an upload to the shared index would put a
stranger's manuscript in someone else's answers — with a citation, and
indistinguishable from a paper that belongs there.

What this Space still cannot claim is "your documents never leave your
machine". A public deployment cannot honestly say that, so it does not. The
version that can runs on your own machine, with no API key:

```bash
git clone https://github.com/asifuddin01/researchlens
cd researchlens
docker compose up
```

Limits here: 20 MB and 80 pages per paper, 5 papers per session, one hour idle.
Text PDFs only — a scan has no text layer to retrieve, and the parser says so
rather than indexing a paper that can never be found.

## Endpoints

| | |
|---|---|
| `GET /health` | corpus size, providers, readiness |
| `GET /library` | the indexed papers, plus your session's |
| `POST /search` | evidence without an answer |
| `POST /ask` | a grounded answer with citations |
| `POST /ask/stream` | the same, as server-sent events |
| `POST /ingest` | add a PDF to your session |
| `DELETE /ingest/{session}` | drop it again |

Source: [github.com/asifuddin01/researchlens](https://github.com/asifuddin01/researchlens)
