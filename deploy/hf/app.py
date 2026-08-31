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


def _generate_on_gpu(system: str, user: str, max_tokens: int = 700) -> str:
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
    # The weights are already on disk by now, so this window covers loading
    # them onto the GPU and generating — not downloading them.
    _generate_on_gpu = spaces.GPU(duration=180)(_generate_on_gpu)
    _prefetch_model()


def _evidence_markdown(citations) -> str:
    if not citations:
        return ""
    rows = ["### Evidence\n"]
    for c in citations:
        # A live result is an abstract fetched from arXiv or PubMed, not a
        # passage from an indexed paper, and is labelled so a reader is never
        # shown one as though it were the other.
        tag = " · **live**" if is_live(c.chunk_id) else ""
        rows.append(
            f"**[{c.marker}]** {c.doc_title}{tag}  \n"
            f"<sub>{c.section_heading} · p{c.pages}</sub>\n\n"
            f"> {c.quote}\n"
        )
    return "\n".join(rows)


def _ask_on_gpu(question: str) -> tuple[str, str, str]:
    """Retrieve with the engine, generate on the attached GPU, then check.

    Retrieval and grounding are the engine's, unchanged — only the call that
    produces text differs. The rules that make an answer trustworthy live
    downstream of the model and apply here identically: the model is never
    invoked without evidence, markers are resolved against the passages that
    were actually retrieved, an invented one is removed, and an answer left
    with nothing supported becomes a refusal.
    """
    started = time.perf_counter()
    evidence, retrieval_ms = ENGINE.retrieve(question)

    if ENGINE.settings and question and _asks_for_survey(question):
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
    return text, _evidence_markdown(citations), timing


def ask(question: str, provider: str) -> tuple[str, str, str]:
    question = (question or "").strip()
    if len(question) < 3:
        return "Ask a question about the indexed papers.", "", ""

    try:
        if provider == "gpu":
            return _ask_on_gpu(question)
        answer = asyncio.run(ENGINE.ask(question, provider))
    except ValueError as e:
        return f"**{e}**", "", ""
    except Exception as e:  # noqa: BLE001 — a demo should explain, not crash
        return f"The model did not answer: {e}", "", ""

    timing = (
        f"{answer.model} · retrieval {answer.retrieval_ms:.0f} ms "
        f"· generation {answer.generation_ms:.0f} ms"
    )
    return answer.text, _evidence_markdown(answer.citations), timing


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
    submit = gr.Button("Ask", variant="primary")

    answer_box = gr.Markdown(label="Answer")
    timing_box = gr.Markdown()
    # Collapsed by default. The evidence is the point, but it is long, and a
    # reader who wants the answer should not have to scroll past five quoted
    # passages to find out whether there was one.
    with gr.Accordion("Evidence", open=True):
        evidence_box = gr.Markdown()

    gr.Examples(examples=[[q] for q in EXAMPLES], inputs=[question])

    submit.click(ask, [question, provider], [answer_box, evidence_box, timing_box])
    question.submit(ask, [question, provider], [answer_box, evidence_box, timing_box])


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
