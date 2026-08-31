"""The HTTP surface.

Deliberately small, and deliberately the same for both deployments. The local
UI and the public page call these endpoints identically; a second API shape for
the demo is the beginning of the second implementation this project is
organised to avoid.

Rate limiting and spend caps are *not* here. They belong at the edge — in the
Cloudflare Worker in front — where rejecting a request costs nothing. Enforcing
them in this process means paying to wake a scale-to-zero container in order to
say no.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from researchlens.config import Settings
from researchlens.engine import Engine
from researchlens.generate.citations import is_grounded, resolve
from researchlens.generate.prompt import NO_EVIDENCE
from researchlens.generate.provider import GenerationRequest


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    provider: str = Field(default="local", pattern="^(local|hosted)$")


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(default=8, ge=1, le=30)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    engine = Engine(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Index at startup, not per request: the ONNX weights and the chunk
        # index are tens of seconds and hundreds of megabytes.
        engine.load()
        yield

    app = FastAPI(
        title="ResearchLens",
        summary="Evidence-grounded retrieval over scientific literature.",
        lifespan=lifespan,
    )

    # The page is served from a different origin than the API on the public
    # deployment. Origins are listed rather than wildcarded so a third-party
    # page cannot spend this instance's budget.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        # A page on https://asifuddin.com reaching an instance on localhost is
        # a public origin calling a private address, which Chrome blocks unless
        # the server opts in. The browser reports ERR_BLOCKED_BY_CLIENT, which
        # looks like an ad blocker and is not one; without this the preflight
        # is refused with "Disallowed CORS private-network" and the real
        # request is never sent.
        #
        # This widens nothing: origin is still checked against the list above,
        # so a page not on it is refused exactly as before.
        allow_private_network=True,
    )

    @app.get("/health")
    async def health():
        providers = []
        for name in ("local", "hosted"):
            try:
                p = engine.provider(name)
            except ValueError:
                continue
            providers.append(
                {"name": name, "model": p.model, "ready": await p.healthy()}
            )
        return {
            "status": "ok" if engine.ready else "loading",
            "mode": settings.mode,
            "papers": len(engine.documents),
            "passages": len(engine.chunks),
            "providers": providers,
        }

    @app.get("/library")
    async def library():
        """The indexed papers, for the source browser."""
        counts: dict[str, int] = {}
        for c in engine.chunks:
            counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
        return {
            "papers": [
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "authors": d.authors,
                    "pages": d.n_pages,
                    "passages": counts.get(d.doc_id, 0),
                }
                for d in sorted(engine.documents, key=lambda d: d.title)
            ]
        }

    @app.post("/search")
    async def search(req: SearchRequest):
        """Evidence without an answer. Useful on its own, and the fastest way
        to tell a retrieval problem from a generation one."""
        hits, ms = engine.retrieve(req.query, top_k=req.k)
        return {
            "query": req.query,
            "retrieval_ms": round(ms, 1),
            "results": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "score": round(r.score, 4),
                    "doc_title": r.chunk.doc_title,
                    "section": r.chunk.section_heading,
                    "pages": r.chunk.pages,
                    "text": r.chunk.text,
                    "found_by": sorted(r.sources),
                }
                for r in hits
            ],
        }

    @app.post("/ask")
    async def ask(req: AskRequest):
        try:
            answer = await engine.ask(req.question, req.provider)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            # The local provider sits behind a container that scales to zero,
            # so an upstream failure is often a cold start rather than a fault.
            # 503 says "try again", which is true; 500 says "broken", which is
            # usually not.
            raise HTTPException(
                status_code=503, detail=f"the {req.provider} model did not answer: {e}"
            ) from e
        return {
            "question": answer.question,
            "text": answer.text,
            "provider": answer.provider,
            "model": answer.model,
            "retrieval_ms": round(answer.retrieval_ms, 1),
            "generation_ms": round(answer.generation_ms, 1),
            "citations": [
                {
                    "marker": c.marker,
                    "chunk_id": c.chunk_id,
                    "doc_title": c.doc_title,
                    "section_heading": c.section_heading,
                    "pages": c.pages,
                    "quote": c.quote,
                }
                for c in answer.citations
            ],
        }

    @app.post("/ask/stream")
    async def ask_stream(req: AskRequest):
        """The same answer, as server-sent events.

        Citations are resolved only once generation completes, because a marker
        cannot be checked against the evidence until it has been written. The
        text streams; the evidence arrives at the end, in one event.
        """
        evidence, retrieval_ms = engine.retrieve(req.question)
        try:
            provider = engine.provider(req.provider)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        async def events():
            if not evidence:
                yield _sse("answer", {"text": NO_EVIDENCE})
                yield _sse("done", {"citations": [], "retrieval_ms": round(retrieval_ms, 1)})
                return

            yield _sse("status", {"retrieval_ms": round(retrieval_ms, 1),
                                  "passages": len(evidence)})
            buf: list[str] = []
            try:
                async for piece in provider.stream(
                    GenerationRequest(question=req.question, evidence=evidence)
                ):
                    buf.append(piece)
                    yield _sse("token", {"t": piece})
            except Exception as e:
                yield _sse("error", {"detail": f"the {req.provider} model stopped: {e}"})
                return

            text, citations = resolve("".join(buf), evidence)
            if not is_grounded(text, citations):
                text, citations = NO_EVIDENCE, []
            yield _sse(
                "done",
                {
                    "text": text,
                    "retrieval_ms": round(retrieval_ms, 1),
                    "citations": [
                        {
                            "marker": c.marker, "chunk_id": c.chunk_id,
                            "doc_title": c.doc_title, "section_heading": c.section_heading,
                            "pages": c.pages, "quote": c.quote,
                        }
                        for c in citations
                    ],
                },
            )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/chunk/{chunk_id:path}")
    async def chunk(chunk_id: str):
        c = engine._by_id.get(chunk_id)
        if c is None:
            raise HTTPException(status_code=404, detail=f"no passage {chunk_id!r}")
        return {
            "chunk_id": c.chunk_id, "doc_id": c.doc_id, "doc_title": c.doc_title,
            "section_heading": c.section_heading, "section_kind": c.section_kind,
            "pages": c.pages, "text": c.text,
        }

    @app.post("/ingest")
    async def ingest():
        if not settings.uploads_enabled:
            raise HTTPException(
                status_code=403,
                detail="this instance serves a fixed corpus; run it locally to add papers",
            )
        raise HTTPException(status_code=501, detail="upload is not implemented yet")

    return app


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
