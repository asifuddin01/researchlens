"""FastAPI surface.

Deliberately small. The demo page and the local UI talk to the same endpoints,
because a second API shape for the public deployment is the beginning of the
second implementation this project is organised to avoid.

Mode is set by RESEARCHLENS_MODE:
  local  — uploads on, no rate limit, full corpus (default)
  demo   — uploads off, fixed corpus, rate limits enforced upstream

The demo's rate limiting and spend cap live in the Cloudflare Worker in front,
not here. Enforcement belongs at the edge, where it costs nothing to reject a
request; doing it in this process means paying to wake a container in order to
say no.
"""

from __future__ import annotations

from researchlens.config import Settings

# Phase III. Endpoint shapes are fixed now because the /researchlens page is
# written against them, and a page built against a guessed contract is a page
# built twice:
#
#   GET  /health          → {status, mode, providers: [{name, model, ready}]}
#   GET  /library         → the indexed papers, for the source browser
#   POST /ask             → {question, provider} → Answer (see types.Answer)
#   POST /ask/stream      → the same, as server-sent events
#   GET  /chunk/{id}      → one passage, for the evidence panel
#   POST /ingest          → local mode only; 403 in demo mode


def create_app(settings: Settings | None = None):
    raise NotImplementedError("Phase III — see docs/phases.md")
