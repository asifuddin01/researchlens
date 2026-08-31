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

This Space is a **read-only exhibit**: a fixed corpus of 101 papers, no
uploads, hosted generation. It is not the system. The system runs on your own
machine with no API key and nothing leaving it:

```bash
git clone https://github.com/asifuddin01/researchlens
cd researchlens
docker compose up
```

A public deployment cannot honestly claim "your documents never leave your
machine", so this one does not.

## Endpoints

| | |
|---|---|
| `GET /health` | corpus size, providers, readiness |
| `GET /library` | the indexed papers |
| `POST /search` | evidence without an answer |
| `POST /ask` | a grounded answer with citations |
| `POST /ask/stream` | the same, as server-sent events |

Source: [github.com/asifuddin01/researchlens](https://github.com/asifuddin01/researchlens)
