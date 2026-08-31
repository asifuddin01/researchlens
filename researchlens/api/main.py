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
import re
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from researchlens.config import Settings
from researchlens.engine import Engine
from researchlens.uploads import MAX_BYTES, UploadError


#: An upload session id. Opaque, client-held, and the only thing separating
#: one reader's uploaded papers from another's — so it is checked for shape
#: rather than trusted: a caller who sends a path or a wildcard here gets a
#: 422, not a lookup.
SESSION = Field(default=None, min_length=8, max_length=64, pattern="^[A-Za-z0-9_-]+$")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    provider: str = Field(default="local", pattern="^(local|hosted)$")
    #: Restrict the answer to these papers. Empty or absent means the whole
    #: corpus, which is the common case and should need no argument.
    doc_ids: list[str] | None = Field(default=None, max_length=200)
    #: Include this session's uploaded papers in the evidence.
    session: str | None = SESSION


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(default=8, ge=1, le=30)
    doc_ids: list[str] | None = Field(default=None, max_length=200)
    session: str | None = SESSION


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
        for name in settings.providers:
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
            "uploads": settings.uploads_enabled,
            "providers": providers,
        }

    @app.get("/library")
    async def library(session: str | None = None):
        """The indexed papers, for the source browser.

        A session's uploaded papers are listed alongside the corpus and marked
        `uploaded`, because the selector that confines an answer to chosen
        papers has to be able to offer them — a paper you cannot pick is a
        paper you cannot ask about on its own.
        """
        counts: dict[str, int] = {}
        for c in engine.chunks:
            counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
        papers = [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "authors": d.authors,
                "pages": d.n_pages,
                "passages": counts.get(d.doc_id, 0),
                "uploaded": False,
            }
            for d in sorted(engine.documents, key=lambda d: d.title)
        ]
        mine = engine.uploaded_documents(session)
        if mine:
            up_counts = engine.uploaded_passage_counts(session)
            papers = [
                {
                    "doc_id": d.doc_id, "title": d.title, "authors": d.authors,
                    "pages": d.n_pages, "passages": up_counts.get(d.doc_id, 0),
                    "uploaded": True,
                }
                for d in mine
            ] + papers
        return {"papers": papers}

    @app.post("/search")
    async def search(req: SearchRequest):
        """Evidence without an answer. Useful on its own, and the fastest way
        to tell a retrieval problem from a generation one."""
        hits, ms = engine.retrieve(
            req.query,
            top_k=req.k,
            doc_ids=set(req.doc_ids) if req.doc_ids else None,
            session=req.session,
        )
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
            answer = await engine.ask(
                req.question,
                req.provider,
                doc_ids=set(req.doc_ids) if req.doc_ids else None,
                session=req.session,
            )
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
        return _answer_json(answer)

    @app.post("/ask/stream")
    async def ask_stream(req: AskRequest):
        """The same answer, as server-sent events.

        A thin mapping over `Engine.ask_stream` rather than its own retrieval.
        This endpoint used to do its own, and had silently drifted: it skipped
        live search and the per-paper cap, so a streamed answer was built from
        different evidence than a waited-for one, with nothing to say which was
        which.
        """
        async def events():
            try:
                stream = engine.ask_stream(
                    req.question,
                    req.provider,
                    doc_ids=set(req.doc_ids) if req.doc_ids else None,
                    session=req.session,
                )
            except ValueError as e:
                yield _sse("error", {"detail": str(e)})
                return
            try:
                async for kind, payload in stream:
                    if kind == "status":
                        yield _sse("status", {
                            "retrieval_ms": round(payload["retrieval_ms"], 1),
                            "passages": payload["passages"],
                            "papers": payload["papers"],
                        })
                    elif kind == "token":
                        yield _sse("token", {"t": payload})
                    elif kind == "error":
                        yield _sse("error", {"detail": payload})
                    elif kind == "done":
                        yield _sse("done", _answer_json(payload))
            except ValueError as e:
                yield _sse("error", {"detail": str(e)})

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
    async def ingest(
        file: UploadFile = File(...),
        session: str | None = Form(default=None),
    ):
        """Add one PDF to the caller's session.

        The paper is parsed, chunked and embedded into an index belonging to
        that session alone. It is never written to disk and never joins the
        corpus — see researchlens/uploads.py for why that is a correctness
        requirement and not caution.

        Returns the session id, generating one when the caller sent none, so a
        first upload needs no setup and every later call can name the same
        session.
        """
        if not settings.uploads_enabled:
            raise HTTPException(
                status_code=403,
                detail="this instance serves a fixed corpus; run it locally to add papers",
            )
        if session is not None and not _SESSION_OK.match(session):
            raise HTTPException(status_code=422, detail="malformed session id")
        session = session or secrets.token_urlsafe(16)

        # Read with a ceiling rather than reading and then measuring: a client
        # can claim any Content-Length it likes, and `await file.read()` on a
        # 2 GB body spends the memory before the check runs.
        raw = await file.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"file exceeds the {MAX_BYTES // 1024 // 1024} MB limit",
            )

        try:
            doc = await run_in_threadpool(
                engine.add_upload, session, raw, file.filename or "upload.pdf"
            )
        except UploadError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return {
            "session": session,
            "doc_id": doc.doc_id,
            "title": doc.title,
            "authors": doc.authors,
            "pages": doc.n_pages,
            "sections": len(doc.sections),
            "papers_open": len(engine.uploaded_documents(session)),
        }

    @app.delete("/ingest/{session}")
    async def forget(session: str):
        """Drop a session's uploaded papers. Idempotent."""
        if not _SESSION_OK.match(session):
            raise HTTPException(status_code=422, detail="malformed session id")
        engine.drop_uploads(session)
        return {"session": session, "papers_open": 0}

    return app


#: Same shape the request models enforce, for the endpoints that take a
#: session in the path or as form data, where pydantic is not doing it.
_SESSION_OK = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _answer_json(answer) -> dict:
    """One shape for an answer, so the streaming and waiting endpoints cannot
    describe the same result differently."""
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


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
