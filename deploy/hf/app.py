"""ResearchLens as a Hugging Face Space.

A Gradio app rather than the Docker image in this directory, because the
Docker SDK is a paid tier and Gradio is the free one. The engine is unchanged —
this is a front end over the same retrieval and the same grounding rules, so
nothing here can answer differently from the local build.

Two things it serves:

  the UI      what a visitor sees on the Space itself, and what the page on
              asifuddin.com embeds
  /health     so that page can tell whether the Space is awake before it
              embeds anything

Generation goes to a hosted OpenAI-compatible endpoint. A free Space has no
persistent volume, so a local model would re-download two gigabytes on every
wake — a minute of waiting, repeated, for a weaker model. Locally the local
model stays the default, where it is the entire point.
"""

from __future__ import annotations

import asyncio
import os
import re
import secrets
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# On a Space this file sits at the repository root beside `researchlens/`, so
# the import works unaided. Run from `deploy/hf/` during development it does
# not, and a file that only runs where it is deployed cannot be tested before
# it is deployed.
for candidate in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[2]):
    if (candidate / "researchlens").is_dir():
        sys.path.insert(0, str(candidate))
        break

import gradio as gr  # noqa: E402

from researchlens.config import Settings  # noqa: E402
from researchlens.engine import Engine  # noqa: E402
from researchlens.generate.citations import is_grounded, resolve  # noqa: E402
from researchlens.generate.prompt import (  # noqa: E402
    NO_EVIDENCE, build_prompt, asks_for_a_survey,
)
from researchlens.generate.provider import GenerationRequest  # noqa: E402
from researchlens.live.arxiv import is_live  # noqa: E402
from researchlens.uploads import (  # noqa: E402
    MAX_PAPERS_PER_SESSION,
    UploadError,
)

# ZeroGPU: a GPU is attached only while a decorated function runs, and a Space
# on that hardware must declare at least one or it refuses to start —
# "No @spaces.GPU function detected during startup", which is exactly how this
# first failed.
#
# Satisfying that with a stub would be a lie told to a scheduler. The honest
# answer is to use the GPU, and it is also the better one: generation on the
# attached A10G needs no API key, which is the claim the whole project rests
# on and which a hosted endpoint quietly breaks.
try:
    import spaces  # type: ignore

    ON_ZERO_GPU = True
except ImportError:
    # Not on a ZeroGPU Space — running locally, or on CPU hardware.
    ON_ZERO_GPU = False

# Twelve rather than thirty. The cross-encoder dominates retrieval time and the
# Space's CPU is much weaker than a laptop's — thirty passes measured 13.6 s
# there against 855 ms locally. Set before the engine is built, because the
# serving configuration is read at import.
os.environ.setdefault("RERANK_CANDIDATES", "12")

ENGINE = Engine(Settings.from_env())
ENGINE.load()

EXAMPLES = [
    "Do deep-learning models outperform linear baselines at predicting perturbation effects?",
    "How does retrieval-augmented generation reduce hallucination?",
    "What datasets are used to benchmark gene regulatory network inference?",
    "What are the current trends in long-context language models?",
]


# The same 3B the local build runs, so the Space and a laptop differ in speed
# rather than in what they say. 7B was the first choice and was wrong for a
# reason worth recording: its weights are about 15 GB, and they were being
# fetched inside the GPU allocation, which would need 125 MB/s sustained to
# finish inside the window. The call died with "connection to the server was
# lost" and the Space stayed healthy, which made it look like a frontend fault.
GPU_MODEL = os.getenv("GPU_MODEL", "Qwen/Qwen2.5-3B-Instruct")

_gpu_pipe = None


def _prefetch_model() -> None:
    """Fetch the weights to disk at startup.

    Downloading needs no GPU, so doing it inside the allocation spends the
    window on network transfer and leaves none for generation. This runs once
    while the Space boots; the GPU call then only has to load from local disk.
    """
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(GPU_MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt"])
        print(f"  weights for {GPU_MODEL} are on disk", flush=True)
    except Exception as e:  # noqa: BLE001
        # Not fatal: the GPU call will fetch them itself, just slowly. Saying
        # so beats a first question that mysteriously times out.
        print(f"  could not prefetch {GPU_MODEL}: {e}", flush=True)


def _load_gpu_model():
    """Move the model onto the GPU. Called inside an allocation, never before.

    ZeroGPU attaches hardware only for the duration of a decorated call, so a
    model loaded at import time would be loaded with no GPU present and stay
    on the CPU for the life of the Space.
    """
    global _gpu_pipe
    if _gpu_pipe is None:
        import torch
        from transformers import pipeline

        _gpu_pipe = pipeline(
            "text-generation",
            model=GPU_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    return _gpu_pipe


def _generate_on_gpu(system: str, user: str, max_tokens: int = 500) -> str:
    pipe = _load_gpu_model()
    out = pipe(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.2,
        return_full_text=False,
    )
    return out[0]["generated_text"].strip()


def _stream_on_gpu(system: str, user: str, max_tokens: int = 500):
    """Generation on the attached GPU, yielded token by token.

    A ZeroGPU allocation covers a generator for as long as it is being drawn
    from, so streaming inside the window is allowed. `TextIteratorStreamer`
    needs the model running on another thread while this one drains the queue —
    the standard transformers arrangement, and the reason `pipe` is called in a
    thread rather than awaited.

    The caller falls back to the blocking path if this raises. Streaming is a
    presentation improvement; being unable to answer at all is not a trade
    worth making for it.
    """
    import threading

    from transformers import TextIteratorStreamer

    pipe = _load_gpu_model()
    streamer = TextIteratorStreamer(
        pipe.tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    worker = threading.Thread(
        target=pipe,
        args=(messages,),
        kwargs={
            "max_new_tokens": max_tokens, "do_sample": True, "temperature": 0.2,
            "return_full_text": False, "streamer": streamer,
        },
        daemon=True,
    )
    worker.start()
    for piece in streamer:
        yield piece
    worker.join(timeout=5)


if ON_ZERO_GPU:
    # 60 seconds, not the 180 first tried. ZeroGPU pads the request — asking
    # for 180 was rejected as "the requested GPU duration (270s) is larger than
    # the maximum allowed", a 1.5x multiplier applied silently. Sixty is the
    # default and is comfortably inside the cap.
    #
    # It is also enough, now that the weights are on disk before the call:
    # what happens inside the window is a load from local storage and a few
    # hundred tokens of generation, not a 6 GB download.
    _generate_on_gpu = spaces.GPU(duration=60)(_generate_on_gpu)
    _stream_on_gpu = spaces.GPU(duration=60)(_stream_on_gpu)
    _prefetch_model()


def _evidence_markdown(citations, mine: set[str] | None = None) -> str:
    if not citations:
        return ""
    mine = mine or set()
    rows = ["### Evidence\n"]
    for c in citations:
        # A live result is an abstract fetched from arXiv or PubMed, not a
        # passage from an indexed paper, and is labelled so a reader is never
        # shown one as though it were the other. A reader's own upload gets the
        # same treatment for the same reason: the three kinds of evidence carry
        # different weight and the answer should not blur them.
        if is_live(c.chunk_id):
            tag = " · **live**"
        elif c.chunk_id.split(":", 1)[0] in mine:
            tag = " · **your upload**"
        else:
            tag = ""
        rows.append(
            f"**[{c.marker}]** {c.doc_title}{tag}  \n"
            f"<sub>{c.section_heading} · p{c.pages}</sub>\n\n"
            f"> {c.quote}\n"
        )
    return "\n".join(rows)


#: A generation fallback that needs no GPU allocation.
#:
#: ZeroGPU is rationed per visitor, and a spent allowance turned the demo into
#: a wall of unsummarised passages — technically the evidence, but not what
#: anyone asked for. Hugging Face's router speaks the OpenAI protocol and takes
#: the same token the Space already authenticates with, so one secret buys a
#: writer that keeps working when the GPU quota does not.
#:
#: Absent by default and absent is fine: without a token this is simply not
#: offered, and the evidence-only path still catches the failure. The project's
#: claim that it runs with no API key is about the *local* build, which is
#: where it matters; a hosted demo borrowing a hosted model breaks nothing.
HF_ROUTER = "https://router.huggingface.co/v1"

#: Deliberately a larger model than the one on the GPU, not a smaller one.
#: A fallback is usually a downgrade; here it is the reverse, because the 3B
#: that fits a ZeroGPU allocation is the weakest link in the whole system —
#: it declines answerable questions and picks the wrong passage out of a good
#: pool. The GPU keeps priority because it costs nothing per call.
#:
#: 72B rather than 7B for a duller reason: checked against the router's own
#: model index, Qwen2.5-7B-Instruct is served by one provider and this by
#: three. A fallback with a single point of failure is not one.
#:
#: Override with FALLBACK_MODEL if the account's inference credits run short —
#: meta-llama/Llama-3.1-8B-Instruct is the cheap end of the same shelf.
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "Qwen/Qwen2.5-72B-Instruct")


def _fallback_provider():
    """A hosted writer, or None when no token is configured."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return None
    from researchlens.generate.openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(
        base_url=HF_ROUTER, api_key=token, model=FALLBACK_MODEL
    )


def _looks_like_quota(err: Exception) -> bool:
    text = str(err).lower()
    return any(
        w in text for w in ("quota", "exceeded", "rate limit", "429", "gpu task aborted")
    )


#: Which model actually wrote the last answer on this path. The timing line
#: used to report GPU_MODEL unconditionally, so an answer written by the
#: fallback was still credited to the 3B — a reader comparing two answers
#: would have been told the same model produced both. Set where the writing
#: happens; read where it is reported.
_LAST_WRITER = GPU_MODEL

#: When the GPU allowance was last found to be spent.
#:
#: Without this, every question pays to discover it again: the allocation is
#: requested, the scheduler refuses it, and only then does the fallback start —
#: measured at 99 s end to end on a question the fallback alone answers in
#: about thirty. A refusal is not per-question, so asking again inside the
#: cooldown learns nothing and costs the reader a minute.
#:
#: Ten minutes because ZeroGPU refills gradually rather than on a published
#: schedule. Guessing low costs one wasted attempt; guessing high means using
#: the fallback while the GPU was already free again.
_GPU_BLOCKED_UNTIL = 0.0
_GPU_COOLDOWN = 600.0


def _stream_from_gpu(question, evidence, history):
    """Generate on the attached GPU, yielding text as it arrives.

    Retrieval and grounding are the engine's, unchanged — only the call that
    produces text differs. The rules that make an answer trustworthy live
    downstream of the model and apply identically here: the model is never
    invoked without evidence, markers are resolved against the passages that
    were actually retrieved, an invented one is removed, and an answer left
    with nothing supported becomes a refusal.
    """
    global _LAST_WRITER, _GPU_BLOCKED_UNTIL

    system, user = build_prompt(question, evidence, history)
    provider = _fallback_provider()

    if provider is not None and time.monotonic() < _GPU_BLOCKED_UNTIL:
        _LAST_WRITER = FALLBACK_MODEL
        yield from _sync_stream(question, evidence, history, provider=provider)
        return

    _LAST_WRITER = GPU_MODEL
    try:
        yield from _stream_on_gpu(system, user)
        return
    except Exception as e:  # noqa: BLE001
        first = e
        if _looks_like_quota(e):
            _GPU_BLOCKED_UNTIL = time.monotonic() + _GPU_COOLDOWN
        else:
            # Streaming is presentation. Losing it should cost the reader a
            # progress bar, not an answer.
            print(f"  GPU streaming failed, one call instead: {e}", flush=True)
            try:
                yield _generate_on_gpu(system, user)
                return
            except Exception as e2:  # noqa: BLE001
                first = e2

    # The allocation itself is unavailable, so retrying it is pointless.
    if provider is None:
        raise first
    print(f"  GPU unavailable ({first}); writing with {FALLBACK_MODEL}", flush=True)
    _LAST_WRITER = FALLBACK_MODEL
    yield from _sync_stream(question, evidence, history, provider=provider)


#: Paper label -> doc_id, for the selector. Titles are truncated because a few
#: run past a hundred characters and a dropdown is not a place to read one.
PAPER_CHOICES = sorted(
    ((f"{d.title[:88]}", d.doc_id) for d in ENGINE.documents),
    key=lambda t: t[0].lower(),
)


def _paper_choices(session: str | None) -> list[tuple[str, str]]:
    """The selector's options: this reader's uploads first, then the corpus.

    Theirs first because a reader who has just added a paper is about to look
    for it, and a hundred and one alphabetised titles is not a place to search.
    """
    mine = [
        (f"\u2b06 {d.title[:80]} (yours)", d.doc_id)
        for d in ENGINE.uploaded_documents(session)
    ]
    return mine + PAPER_CHOICES


def forget(session: str | None) -> tuple[str, str, dict]:
    """Drop this reader's papers now rather than at the timeout."""
    if session:
        ENGINE.drop_uploads(session)
    return (
        session,
        "Your papers have been dropped.",
        gr.update(choices=_paper_choices(None), value=[]),
    )


def _pairs(messages, limit: int = 1):
    """The last few (question, answer) turns, for pronoun resolution only.

    Deliberately short. The prompt carries this so "what about the second one"
    can be understood, not so the model can answer from what it said earlier —
    an answer built on a previous answer is an answer with no passage behind
    it, which is the one thing this system exists to prevent.
    """
    out, pending = [], None
    for m in messages or []:
        role = m.get("role") if isinstance(m, dict) else None
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, str):
            # A multimodal turn's content is a file tuple, not text. It carries
            # no question to resolve a pronoun against.
            continue
        if role == "user":
            pending = content
        elif role == "assistant" and pending is not None:
            out.append((pending, content))
            pending = None
    return out[-limit:]


def _sync_stream(question, evidence, history, provider_name=None, provider=None):
    """Drain an async provider stream from synchronous Gradio code.

    Gradio's handler is a plain generator and the providers are async. Running
    the coroutine to completion first would defeat the point, so the loop is
    stepped one chunk at a time and each is handed back as it lands.
    """
    provider = provider or ENGINE.provider(provider_name)
    req = GenerationRequest(question=question, evidence=evidence, history=history)
    loop = asyncio.new_event_loop()
    try:
        agen = provider.stream(req).__aiter__()
        while True:
            try:
                yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                return
    finally:
        loop.close()


def respond(message, history, provider, papers, live_mode, session):
    """One conversational turn: ingest anything attached, then answer.

    A generator, so the reader watches retrieval finish and the answer being
    written rather than a spinner and a wall of text. Yields
    (reply, evidence, timing, session, paper choices) throughout.
    """
    session = session or secrets.token_urlsafe(16)
    files = list((message or {}).get("files") or [])
    question = ((message or {}).get("text") or "").strip()
    choices = gr.update()

    # ---- anything dropped into the box is a paper to read ----------------
    notes: list[str] = []
    if files:
        yield (f"_Reading {len(files)} file(s)…_", "", "", session, choices)
        for f in files:
            path = Path(getattr(f, "name", f))
            try:
                doc = ENGINE.add_upload(session, path.read_bytes(), path.name)
                notes.append(f"Indexed **{doc.title or path.name}** — {doc.n_pages} pages.")
            except UploadError as e:
                notes.append(f"\u26a0 {path.name}: {e}")
            except Exception as e:  # noqa: BLE001 — a demo should explain, not crash
                notes.append(f"\u26a0 {path.name}: could not be read ({e})")
        choices = gr.update(choices=_paper_choices(session))
        open_now = len(ENGINE.uploaded_documents(session))
        if open_now:
            notes.append(
                f"<sub>{open_now}/{MAX_PAPERS_PER_SESSION} of your papers open · "
                "held in memory for this session only, never added to the public "
                "corpus, never written to disk.</sub>"
            )
        if not question:
            # Papers with no question: say what was read and stop. Inventing a
            # question to answer would be answering something nobody asked.
            yield ("\n\n".join(notes) + "\n\nAsk me about them.", "", "", session, choices)
            return

    if len(question) < 3:
        yield ("Ask a question about the papers.", "", "", session, choices)
        return

    preamble = ("\n\n".join(notes) + "\n\n---\n\n") if notes else ""
    history_pairs = _pairs(history)
    yield (preamble + "_searching the papers…_", "", "", session, choices)

    doc_ids = set(papers) if papers else None
    live = {"auto": None, "always": True, "never": False}[live_mode]
    mine = {d.doc_id for d in ENGINE.uploaded_documents(session)}

    try:
        evidence, retrieval_ms = asyncio.run(
            ENGINE.evidence_for(question, live=live, doc_ids=doc_ids, session=session)
        )
    except Exception as e:  # noqa: BLE001
        yield (preamble + f"Retrieval failed: {e}", "", "", session, choices)
        return

    if not evidence:
        yield (preamble + NO_EVIDENCE, "",
               f"retrieval {retrieval_ms:.0f} ms · no passages", session, choices)
        return

    n_live = sum(1 for r in evidence if is_live(r.chunk.chunk_id))
    n_mine = sum(1 for r in evidence if r.chunk.doc_id in mine)
    found = (
        f"{len(evidence)} passages from {len({r.chunk.doc_id for r in evidence})} papers"
        + (f" · {n_live} live" if n_live else "")
        + (f" · {n_mine} yours" if n_mine else "")
    )
    status = f"retrieval {retrieval_ms:.0f} ms · {found}"
    yield (preamble + f"_{found} — writing the answer…_", "", status, session, choices)

    started = time.perf_counter()
    buf: list[str] = []
    try:
        pieces = (
            _stream_from_gpu(question, evidence, history_pairs)
            if provider == "gpu"
            else _sync_stream(question, evidence, history_pairs, provider)
        )
        for piece in pieces:
            buf.append(piece)
            # Streamed straight through. Markers are resolved only at the end,
            # so text shown mid-flight may contain one that does not survive —
            # the alternative is a blank box for twenty seconds, and the final
            # replacement is what the reader keeps.
            yield (preamble + "".join(buf), "", status, session, choices)
    except Exception as e:  # noqa: BLE001
        yield (preamble + f"The model did not answer: {e}", "", status, session, choices)
        return
    generation_ms = (time.perf_counter() - started) * 1000.0

    text, citations = resolve("".join(buf), evidence)
    if not is_grounded(text, citations):
        # Every marker was invented, or none was produced. The reader has
        # already watched the text appear and it is still replaced: a
        # retraction is recoverable, an unsupported claim shown as sourced is
        # not.
        text, citations = NO_EVIDENCE, []

    model = _LAST_WRITER if provider == "gpu" else ENGINE.provider(provider).model
    timing = f"{model} · retrieval {retrieval_ms:.0f} ms · generation {generation_ms:.0f} ms · {found}"
    yield (preamble + text, _evidence_markdown(citations, mine), timing, session, choices)


def corpus_stats() -> dict:
    """What the running system actually holds, so a page never states a figure
    the system disagrees with."""
    return {
        "papers": len(ENGINE.documents),
        "passages": len(ENGINE.chunks),
        "model": GPU_MODEL if ON_ZERO_GPU else ENGINE.settings.ollama_model,
        "sources": ["arxiv", "pubmed", "openalex"],
    }


def add_paper(file_path: str, session: str = "") -> dict:
    """Index one uploaded PDF for a caller's session, and say what it was.

    The website's own ask panel calls this; the Space's chat box does the same
    thing through its message box. Both land in the same per-session index,
    never written to disk and never merged into the public corpus — see
    researchlens/uploads.py for why that is a correctness requirement rather
    than caution.
    """
    session = session or secrets.token_urlsafe(16)
    if not file_path:
        return {"error": "No file was sent.", "session": session}

    path = Path(file_path)
    try:
        doc = ENGINE.add_upload(session, path.read_bytes(), path.name)
    except UploadError as e:
        return {"error": str(e), "session": session}
    except Exception as e:  # noqa: BLE001 — a demo should explain, not crash
        return {"error": f"{path.name} could not be read: {e}", "session": session}

    return {
        "session": session,
        "doc_id": doc.doc_id,
        "title": doc.title or path.name,
        "pages": doc.n_pages,
        "papers_open": len(ENGINE.uploaded_documents(session)),
        "limit": MAX_PAPERS_PER_SESSION,
    }


def forget_papers(session: str = "") -> dict:
    """Drop a caller's uploaded papers now rather than at the timeout."""
    if session:
        ENGINE.drop_uploads(session)
    return {"session": session, "papers_open": 0}


def my_papers(session: str = "") -> dict:
    """What this caller currently has open.

    Exists because an uploaded paper was invisible after the message about it
    scrolled away, and a reader could not tell whether it was still there. A
    page that shows a list can answer that without asking.
    """
    docs = ENGINE.uploaded_documents(session or None)
    return {
        "session": session,
        "papers": [
            {"doc_id": d.doc_id, "title": d.title, "pages": d.n_pages} for d in docs
        ],
        "limit": MAX_PAPERS_PER_SESSION,
    }


def find_similar(session: str = "", doc_id: str = "", max_results: int = 9) -> dict:
    """Recent work related to a paper the reader added.

    The engine owns this; the two surfaces must not each have their own query
    ladder. Abstracts, not full text, and labelled as such wherever they
    appear — a paper found this way is a lead, not a finding.
    """
    try:
        return asyncio.run(
            ENGINE.similar_papers(session or None, doc_id, max_results)
        )
    except Exception as e:  # noqa: BLE001 — a demo should explain, not crash
        return {"error": f"Literature search failed: {e}"}


def ask_json(
    question: str,
    provider: str = "",
    papers: list[str] | None = None,
    live: str = "auto",
    session: str = "",
) -> dict:
    """One answer, with its citations, as data.

    The same engine, the same evidence and the same grounding rules the UI
    uses — this differs only in returning records instead of markdown, so the
    two surfaces cannot answer differently.
    """
    question = (question or "").strip()
    if len(question) < 3:
        return {"error": "Ask a question about the papers."}

    provider = provider or (PROVIDERS[0] if PROVIDERS else "hosted")
    doc_ids = set(papers) if papers else None
    want_live = {"auto": None, "always": True, "never": False}.get(live)

    try:
        evidence, retrieval_ms = asyncio.run(
            ENGINE.evidence_for(
                question, live=want_live, doc_ids=doc_ids, session=session or None
            )
        )
    except Exception as e:  # noqa: BLE001 — a demo should explain, not crash
        return {"error": f"Retrieval failed: {e}"}

    if not evidence:
        return {
            "text": NO_EVIDENCE, "citations": [], "model": "",
            "retrieval_ms": round(retrieval_ms, 1), "generation_ms": 0.0,
            "passages": 0, "papers": 0,
        }

    mine = {d.doc_id for d in ENGINE.uploaded_documents(session or None)}
    # Where a passage can be opened. Only web-fetched sources carry one; a
    # corpus PDF is cited by page instead. Built from the evidence because
    # `Citation` records the marker and the quote, not the origin URL.
    links = {r.chunk.chunk_id: r.chunk.url for r in evidence if r.chunk.url}
    from_site = [r for r in evidence if r.chunk.chunk_id.startswith("site:")]
    # A recency question answered from a fixed corpus, with the reader never
    # told that Local is why. The prompt already makes the model add a clause
    # about it, which is buried in the prose and says nothing about what to do.
    # Reported as a flag so the page can offer the fix instead.
    stranded = (
        asks_for_a_survey(question)
        and want_live is False
        and not any(is_live(r.chunk.chunk_id) for r in evidence)
    )

    started = time.perf_counter()
    try:
        if provider == "gpu":
            raw = "".join(_stream_from_gpu(question, evidence, None))
        else:
            raw = "".join(_sync_stream(question, evidence, None, provider))
    except Exception as e:  # noqa: BLE001
        # Retrieval needs no GPU. When only the writer is unavailable — a spent
        # ZeroGPU quota is the usual reason — the evidence is still there and
        # is most of what this system is for. Returning it is both more useful
        # than an error and more honest than a generated answer would have
        # been: these are the passages, unsummarised, with nothing between the
        # reader and the source.
        return {
            "text": "",
            "evidence_only": True,
            "reason": str(e),
            "citations": [
                {
                    "marker": i,
                    "chunk_id": r.chunk.chunk_id,
                    "doc_title": r.chunk.doc_title,
                    "section_heading": r.chunk.section_heading,
                    "pages": r.chunk.pages,
                    "quote": r.chunk.text[:400],
                    "live": is_live(r.chunk.chunk_id),
                    "yours": r.chunk.doc_id in mine,
                    "site": r.chunk.chunk_id.startswith("site:"),
                    "url": r.chunk.url or "",
                }
                for i, r in enumerate(evidence, start=1)
            ],
            "model": "",
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": 0.0,
            "passages": len(evidence),
            "papers": len({r.chunk.doc_id for r in evidence}),
            "about_author": bool(from_site),
            "asks_current_but_local": stranded,
        }
    generation_ms = (time.perf_counter() - started) * 1000.0

    text, citations = resolve(raw, evidence)
    if not is_grounded(text, citations):
        text, citations = NO_EVIDENCE, []

    return {
        "text": text,
        "citations": [
            {
                "marker": c.marker,
                "chunk_id": c.chunk_id,
                "doc_title": c.doc_title,
                "section_heading": c.section_heading,
                "pages": c.pages,
                "quote": c.quote,
                "live": is_live(c.chunk_id),
                "yours": c.chunk_id.split(":", 1)[0] in mine,
                "site": c.chunk_id.startswith("site:"),
                "url": links.get(c.chunk_id, ""),
            }
            for c in citations
        ],
        "model": _LAST_WRITER if provider == "gpu" else ENGINE.provider(provider).model,
        "retrieval_ms": round(retrieval_ms, 1),
        "generation_ms": round(generation_ms, 1),
        "passages": len(evidence),
        "papers": len({r.chunk.doc_id for r in evidence}),
        # So the page can say the answer came from the site rather than the
        # literature. A reader deserves to know a self-description is what
        # they are reading before they decide how much to trust it.
        "about_author": bool(from_site),
        "asks_current_but_local": stranded,
    }


# On ZeroGPU the attached GPU is the best option and needs no key, so it leads.
# Anything else the instance has configured follows it.
PROVIDERS = (["gpu"] if ON_ZERO_GPU else []) + [
    p for p in ENGINE.settings.providers
    if p != "local" or not ON_ZERO_GPU
]
if not PROVIDERS:
    PROVIDERS = ["hosted"]

with gr.Blocks(title="ResearchLens", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
# ResearchLens

Ask across **{len(ENGINE.documents)} papers** ({len(ENGINE.chunks):,} passages)
in single-cell genomics, causal discovery, vision-language models and
retrieval-augmented generation — or attach your own PDF and ask about that.

It answers only from evidence. Every claim resolves to a passage: a paper, a
section, a page. When nothing supports an answer it says so instead of writing
one. Questions about what is *current* also search arXiv, PubMed and OpenAlex;
those are marked **live** and are abstracts, not full text.

Runs locally with no API key —
[github.com/asifuddin01/researchlens](https://github.com/asifuddin01/researchlens).
"""
    )

    # Built with render=False and rendered below by hand, so the evidence sits
    # under the conversation and the controls under that. ChatInterface renders
    # anything it is handed that has not been rendered already, which would put
    # all of it above the chat.
    session = gr.State(value=None)
    evidence_box = gr.Markdown(render=False)
    timing_box = gr.Markdown(render=False)
    papers = gr.Dropdown(
        choices=PAPER_CHOICES,
        value=[],
        multiselect=True,
        label=f"Papers (leave empty to search all {len(ENGINE.documents)})",
        info="Pick one or more to confine the answer to them.",
        render=False,
    )
    provider = gr.Radio(
        choices=PROVIDERS,
        value=PROVIDERS[0] if PROVIDERS else None,
        label="Model",
        render=False,
    )
    live_mode = gr.Radio(
        choices=["auto", "always", "never"],
        value="auto",
        label="Search arXiv, PubMed and OpenAlex",
        info="auto: only when the question asks what a field is doing now — a "
             "fixed corpus cannot answer that.",
        render=False,
    )

    # gr.ChatInterface rather than a Chatbot wired up by hand: it brings stop,
    # retry, undo and message editing, none of which is worth reimplementing,
    # and it is the component Gradio actually maintains for this shape of app.
    #
    # A note against a wasted hour: synthetic Enter keystrokes from browser
    # automation do not submit this box, and neither did the hand-wired
    # Textbox before it. Two components failing the same way points at the
    # instrument, not at Gradio — dispatched key events reach the DOM but not
    # Svelte's handler. Click the send button when driving this from a script.
    #
    # multimodal=True puts the file picker inside the message box, which is the
    # right place for it: attaching a paper and asking about it is one action,
    # not a trip to a separate upload panel.
    gr.ChatInterface(
        fn=respond,
        multimodal=True,
        textbox=gr.MultimodalTextbox(
            placeholder="Ask about the papers, or attach a PDF and ask about that…",
            file_types=[".pdf"],
            file_count="multiple",
            show_label=False,
            autofocus=True,
        ),
        chatbot=gr.Chatbot(
            height=460,
            label="Conversation",
            placeholder="Ask about the indexed papers, or attach your own.",
            render=False,
        ),
        additional_inputs=[provider, papers, live_mode, session],
        # ChatInterface renders the additional inputs itself, in this
        # accordion. Rendering them again below raises DuplicateBlockError —
        # a component belongs to exactly one place in the layout.
        # render=False, or it appears twice: creating an Accordion inside a
        # Blocks context registers it in the layout at that point, and
        # ChatInterface then renders it again around the inputs.
        additional_inputs_accordion=gr.Accordion(
            "Scope and model", open=False, render=False
        ),
        additional_outputs=[evidence_box, timing_box, session, papers],
        examples=[[{"text": q, "files": []}] for q in EXAMPLES],
        editable=True,
        save_history=False,
    )

    # A JSON endpoint beside the UI, for asifuddin.com to render in its own
    # design. The page had been embedding this Space in an iframe, which meant
    # a reader met two visual languages at once — the site's, then Gradio's
    # inside a box in the middle of it. Handing over structured citations lets
    # the page draw the answer the way it draws everything else, and the Space
    # stays what it is for anyone who visits it directly.
    #
    # gr.api rather than a route on demo.app: Gradio builds its FastAPI app
    # inside launch(), so a route decorated onto demo.app beforehand is
    # silently discarded — that is how an earlier /health returned 404 while
    # the app served fine.
    gr.api(ask_json, api_name="ask_json")
    gr.api(corpus_stats, api_name="corpus_stats")
    gr.api(add_paper, api_name="add_paper")
    gr.api(forget_papers, api_name="forget_papers")
    gr.api(my_papers, api_name="my_papers")
    gr.api(find_similar, api_name="find_similar")

    timing_box.render()
    with gr.Accordion("Evidence", open=True):
        evidence_box.render()

    with gr.Accordion("Your own papers", open=False):
        gr.Markdown(
            f"""
**Your own papers.** Attach up to **{MAX_PAPERS_PER_SESSION} PDFs** in the
message box and ask about them — alone, or against the
{len(ENGINE.documents)} indexed ones.

They are parsed, chunked and embedded in memory for **your session only**.
Nothing is written to disk, nothing joins the public corpus, nothing is visible
to anyone else, and an idle session is dropped after an hour.

Text PDFs only: a scan has no text layer to retrieve, and the parser says so
rather than indexing a paper that can never be found.
"""
        )
        forget_btn = gr.Button("Forget my papers", size="sm")
        forget_note = gr.Markdown()
        forget_btn.click(forget, [session], [session, forget_note, papers])


# No custom /health here, deliberately. Gradio builds its FastAPI app inside
# launch(), so a route decorated onto `demo.app` beforehand is silently
# discarded — the endpoint returned 404 while the app served fine, which is
# the worst kind of broken.
#
# Nothing needs it either: the portfolio page embeds this Space in an iframe
# rather than probing it, and the probe it does run is for an instance on the
# reader's own machine. Gradio's own /config answers "is it up".

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
