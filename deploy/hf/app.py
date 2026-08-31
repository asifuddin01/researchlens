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
import warnings

warnings.filterwarnings("ignore")

import gradio as gr  # noqa: E402

from researchlens.config import Settings  # noqa: E402
from researchlens.engine import Engine  # noqa: E402
from researchlens.live.arxiv import is_live  # noqa: E402

ENGINE = Engine(Settings.from_env())
ENGINE.load()

EXAMPLES = [
    "Do deep-learning models outperform linear baselines at predicting perturbation effects?",
    "How does retrieval-augmented generation reduce hallucination?",
    "What datasets are used to benchmark gene regulatory network inference?",
    "What are the current trends in long-context language models?",
]


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


def ask(question: str, provider: str) -> tuple[str, str, str]:
    question = (question or "").strip()
    if len(question) < 3:
        return "Ask a question about the indexed papers.", "", ""
    try:
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


PROVIDERS = list(ENGINE.settings.providers)

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
    evidence_box = gr.Markdown()

    gr.Examples(examples=[[q] for q in EXAMPLES], inputs=[question])

    submit.click(ask, [question, provider], [answer_box, evidence_box, timing_box])
    question.submit(ask, [question, provider], [answer_box, evidence_box, timing_box])


# Gradio is built on FastAPI, so the health endpoint the portfolio page probes
# can be added to the same app rather than needing a second service.
@demo.app.get("/health")
def health():
    return {
        "status": "ok" if ENGINE.ready else "loading",
        "mode": ENGINE.settings.mode,
        "papers": len(ENGINE.documents),
        "passages": len(ENGINE.chunks),
        "providers": [{"name": p, "model": "", "ready": True} for p in PROVIDERS],
    }


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
