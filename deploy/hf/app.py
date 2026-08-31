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
    NO_EVIDENCE,
    asks_for_a_survey as _asks_for_survey,
    build_prompt,
)
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


def _ask_on_gpu(
    question: str, doc_ids: set[str] | None = None, session: str | None = None
) -> tuple[str, str, str]:
    """Retrieve with the engine, generate on the attached GPU, then check.

    Retrieval and grounding are the engine's, unchanged — only the call that
    produces text differs. The rules that make an answer trustworthy live
    downstream of the model and apply here identically: the model is never
    invoked without evidence, markers are resolved against the passages that
    were actually retrieved, an invented one is removed, and an answer left
    with nothing supported becomes a refusal.
    """
    started = time.perf_counter()
    evidence, retrieval_ms = ENGINE.retrieve(question, doc_ids=doc_ids, session=session)
    mine = {d.doc_id for d in ENGINE.uploaded_documents(session)}

    # Not when a subset is chosen: a reader who picked two papers has said the
    # rest of the literature is not what they want. Not when they have uploaded
    # a paper either — they came to ask about that.
    if not doc_ids and not mine and question and _asks_for_survey(question):
        fetched = asyncio.run(ENGINE.live_evidence(question))
        if fetched:
            evidence = ENGINE._merge_live(question, fetched, evidence)

    if not evidence:
        return NO_EVIDENCE, "", f"retrieval {retrieval_ms:.0f} ms"

    system, user = build_prompt(question, evidence)
    gen_started = time.perf_counter()
    raw = _generate_on_gpu(system, user)
    generation_ms = (time.perf_counter() - gen_started) * 1000.0

    text, citations = resolve(raw, evidence)
    if not is_grounded(text, citations):
        text, citations = NO_EVIDENCE, []

    timing = (
        f"{GPU_MODEL} on ZeroGPU · retrieval {retrieval_ms:.0f} ms "
        f"· generation {generation_ms:.0f} ms"
    )
    _ = time.perf_counter() - started
    return text, _evidence_markdown(citations, mine), timing


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


def upload(
    files, session: str | None, progress=gr.Progress()
) -> tuple[str, str, dict]:
    """Index the reader's PDFs into a session of their own.

    Nothing is written to disk and nothing joins the corpus — the papers live
    in this browser session and are dropped when it goes idle. That is stated
    in the UI rather than buried here, because a reader handing over an
    unpublished manuscript is owed the answer to "where does this go".
    """
    session = session or secrets.token_urlsafe(16)
    if not files:
        return session, "", gr.update()

    added, failed = [], []
    for i, f in enumerate(files):
        path = Path(getattr(f, "name", f))
        # Indexing a sixteen-page paper is about seven seconds on a laptop and
        # longer on this Space's CPU: parse, chunk, then embed every passage.
        # Without a progress line that reads as a page that has stopped
        # working, and the reader uploads it again.
        progress(i / max(len(files), 1), desc=f"Reading {path.name}…")
        try:
            doc = ENGINE.add_upload(session, path.read_bytes(), path.name)
            added.append(f"**{doc.title or path.name}** — {doc.n_pages} pages")
        except UploadError as e:
            failed.append(f"{path.name}: {e}")
        except Exception as e:  # noqa: BLE001 — a demo should explain, not crash
            failed.append(f"{path.name}: could not be read ({e})")

    lines = []
    if added:
        lines.append("Indexed " + "; ".join(added) + ".")
        lines.append(
            "Ask anything — your papers get guaranteed room in the evidence, "
            "or pick them in the selector to read them alone."
        )
    for msg in failed:
        lines.append(f"\u26a0 {msg}")
    open_now = len(ENGINE.uploaded_documents(session))
    if open_now:
        lines.append(
            f"<sub>{open_now}/{MAX_PAPERS_PER_SESSION} open · held in memory for "
            "this session only, never added to the public corpus, never written "
            "to disk.</sub>"
        )
    return session, "\n\n".join(lines), gr.update(choices=_paper_choices(session))


def forget(session: str | None) -> tuple[str, str, dict, None]:
    """Drop this reader's papers now rather than at the timeout."""
    if session:
        ENGINE.drop_uploads(session)
    return session, "Your papers have been dropped.", gr.update(
        choices=_paper_choices(None), value=[]
    ), None


def ask(
    question: str,
    provider: str,
    papers: list[str] | None = None,
    session: str | None = None,
) -> tuple[str, str, str]:
    question = (question or "").strip()
    if len(question) < 3:
        return "Ask a question about the indexed papers.", "", ""

    # An empty selection means the whole corpus, which is the common case and
    # should not require choosing a hundred and one things.
    doc_ids = set(papers) if papers else None

    try:
        if provider == "gpu":
            return _ask_on_gpu(question, doc_ids, session)
        answer = asyncio.run(
            ENGINE.ask(question, provider, doc_ids=doc_ids, session=session)
        )
    except ValueError as e:
        return f"**{e}**", "", ""
    except Exception as e:  # noqa: BLE001 — a demo should explain, not crash
        return f"The model did not answer: {e}", "", ""

    timing = (
        f"{answer.model} · retrieval {answer.retrieval_ms:.0f} ms "
        f"· generation {answer.generation_ms:.0f} ms"
    )
    mine = {d.doc_id for d in ENGINE.uploaded_documents(session)}
    return answer.text, _evidence_markdown(answer.citations, mine), timing


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

Evidence-grounded retrieval over **{len(ENGINE.documents)} papers**
({len(ENGINE.chunks):,} passages) in single-cell genomics, causal discovery,
vision-language models and retrieval-augmented generation.

Every claim resolves to a passage — a paper, a section, a page. Questions about
what is *current* also search arXiv and PubMed; those results are marked
**live** and are abstracts, not full text.

This is a read-only exhibit of a system that runs locally with no API key —
[github.com/asifuddin01/researchlens](https://github.com/asifuddin01/researchlens).
"""
    )

    with gr.Row():
        question = gr.Textbox(
            label="Your question",
            placeholder="What does the literature say about…",
            # Without an explicit height a Textbox expands to fill its row,
            # which left a question box taller than the answer beneath it.
            lines=2,
            max_lines=4,
            scale=5,
        )
        provider = gr.Radio(
            choices=PROVIDERS,
            value=PROVIDERS[0] if PROVIDERS else None,
            label="Model",
            scale=1,
        )
    papers = gr.Dropdown(
        choices=PAPER_CHOICES,
        value=[],
        multiselect=True,
        label=f"Papers (leave empty to search all {len(ENGINE.documents)})",
        info="Pick one or more to confine the answer to them.",
    )
    submit = gr.Button("Ask", variant="primary")

    # A session id, held per browser tab by Gradio. It is the only thing
    # separating one reader's uploaded papers from another's, so it is
    # generated here rather than accepted from anywhere.
    session = gr.State(value=None)

    with gr.Accordion("Add your own papers", open=False):
        gr.Markdown(
            f"""
Drop in up to **{MAX_PAPERS_PER_SESSION} PDFs** and ask about them — on their
own, or against the {len(ENGINE.documents)} indexed papers.

They are parsed, chunked and embedded in memory for **your session only**.
Nothing is written to disk, nothing is added to the public corpus, and nothing
is visible to anyone else. An idle session is dropped after an hour.

Text PDFs only: a scan has no text layer to retrieve, and the parser will say so
rather than index a paper that can never be found.
"""
        )
        uploader = gr.File(
            label="PDFs",
            file_count="multiple",
            file_types=[".pdf"],
            height=140,
        )
        upload_status = gr.Markdown()
        clear_uploads = gr.Button("Forget my papers", size="sm")

    answer_box = gr.Markdown(label="Answer")
    timing_box = gr.Markdown()
    # Collapsed by default. The evidence is the point, but it is long, and a
    # reader who wants the answer should not have to scroll past five quoted
    # passages to find out whether there was one.
    with gr.Accordion("Evidence", open=True):
        evidence_box = gr.Markdown()

    gr.Examples(examples=[[q] for q in EXAMPLES], inputs=[question])

    inputs = [question, provider, papers, session]
    outputs = [answer_box, evidence_box, timing_box]
    submit.click(ask, inputs, outputs)
    question.submit(ask, inputs, outputs)

    uploader.upload(
        upload, [uploader, session], [session, upload_status, papers]
    )
    clear_uploads.click(
        forget, [session], [session, upload_status, papers, uploader]
    )


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
