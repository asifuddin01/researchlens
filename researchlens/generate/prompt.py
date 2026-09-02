"""Building the prompt, and checking what comes back.

Grounding is not an instruction in the prompt — a 3B model will politely agree
to cite its sources and then not. It is enforced mechanically, here and in
`citations.py`:

1. The model only ever sees passages retrieved from the corpus.
2. Every passage is numbered, and the instruction is to cite by number.
3. A marker pointing at a passage that was never retrieved is stripped after
   generation, so an invented [7] cannot reach the reader.
4. An answer left with no supported citation is reported as no-evidence.
5. When retrieval returns nothing, the model is never called at all.

Rule 5 matters most. A model asked a question with no context will answer from
its training, fluently, and a fluent unsourced answer is worse than a refusal
because the two are indistinguishable to a reader.
"""

from __future__ import annotations

from researchlens.live.arxiv import is_live
from researchlens.live.author import asks_about_the_author
from researchlens.live.query import asks_for_limitations
from researchlens.types import Retrieved

SYSTEM = """You answer questions using only the numbered passages provided.

Rules:
- Use only the passages. If they do not answer the question, say so plainly.
- Cite every claim with the passage number in square brackets, like [2].
- A sentence stating a fact from the passages must carry a citation.
- Do not add background knowledge, even if you are confident it is correct.
- Quote exact numbers, dataset names and model names as they appear.
- Be concise. Three or four sentences is usually enough.
- A passage marked (ABSTRACT ONLY) is a paper's abstract from a live search,
  not its full text. It supports what that paper claims or sets out to do. It
  does not support a statement about what was measured, on which dataset, or
  with what result, because the abstract may not say and a reader cannot check.
- A passage marked (AUTHOR'S OWN WEBSITE) is the author describing their own
  work. Answer questions about the author from these, cited like any other.
  It is a self-description, not a reviewed finding. The papers in the other
  passages are ones the author has read, not ones they wrote.
- A passage marked (AUTHOR'S TEXTBOOK) is the author's own explanation of a
  topic. Use it to explain, and cite it. It is teaching, not evidence: it does
  not report a result, and no number in it is a measurement of anything.

If the question asks what is *current*, *recent*, *trending*, or what a field
is doing now, you are being asked something a fixed set of papers cannot fully
answer. Say what these particular papers show, name that limit in one clause,
and do not present the passages as a survey of the field."""

NO_EVIDENCE = (
    "I could not find sufficient evidence in the indexed papers to answer that."
)


def _passage_kind(chunk_id: str) -> str:
    """What to call this block so the model does not overstate it.

    A bug worth recording: this test used to be `startswith("arxiv:")`, inline
    and written when arXiv was the only live source. PubMed and OpenAlex were
    added later, and their abstracts have been reaching the model labelled
    "passage" ever since — the precise confusion the label exists to prevent,
    in two of the three live sources, while `is_live` twenty lines below had
    the correct list all along. Asking the source of truth rather than keeping
    a second copy of it is what fixes it and what stops it recurring.
    """
    if is_live(chunk_id):
        return "ABSTRACT ONLY"
    # Not a paper at all. The model must not write about a personal website as
    # though it reported, measured or found anything.
    if chunk_id.startswith("site:"):
        return "AUTHOR'S OWN WEBSITE"
    # A proposition from the author's textbook: an explanation he wrote, not a
    # result anybody measured.
    if chunk_id.startswith("elementa:"):
        return "AUTHOR'S TEXTBOOK"
    return "passage"


def build_context(evidence: list[Retrieved], max_chars: int = 9000) -> str:
    """Number the passages and label each with where it came from.

    The source line is part of the passage on purpose: the model is more
    consistent about citing a block that visibly belongs to a paper than one
    that appears as anonymous text, and the same label is what the citation
    resolver reads back.

    Passages are truncated as a whole rather than individually, so a passage
    that appears is always complete — half a passage can be cited for a claim
    its missing half contradicts.
    """
    parts: list[str] = []
    used = 0
    for i, r in enumerate(evidence, start=1):
        c = r.chunk
        # Live results are abstracts fetched from a search API, not passages
        # from an indexed paper. Marking them keeps the model from writing
        # "the paper reports X on dataset Y" when the abstract never said so —
        # an abstract supports what a paper *claims*, not what it measured.
        kind = _passage_kind(c.chunk_id)
        block = (
            f"[{i}] ({kind}) {c.doc_title} — {c.section_heading}, {c.page_ref}\n{c.text}\n"
        )
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


_SURVEY_WORDS = (
    "current", "recent", "trend", "trends", "trending", "nowadays", "lately",
    "state of the art", "state-of-the-art", "emerging", "latest", "these days",
)


def asks_for_a_survey(question: str) -> bool:
    """Whether the question asks about a field rather than about the papers.

    "What are the current trends in long-context language models?" cannot be
    answered from a fixed corpus, and answering it anyway is the failure that
    looks most like success: the model writes a fluent survey from what it
    already believed, decorated with whatever citations retrieval happened to
    return. Observed directly — the system answered a question about current
    LLM trends from *Attention Is All You Need* (2017) and BERT (2018), with
    real citations and no indication that its evidence was eight years old.
    """
    q = question.lower()
    return any(w in q for w in _SURVEY_WORDS)


def build_prompt(question: str, evidence: list[Retrieved], history=None) -> tuple[str, str]:
    """Return (system, user). Empty evidence is the caller's error to avoid."""
    if not evidence:
        raise ValueError(
            "build_prompt called with no evidence — the model must not be "
            "invoked at all when retrieval returns nothing (rule 5)"
        )

    context = build_context(evidence)
    turns = ""
    if history:
        # Only the last exchange, and only to resolve pronouns. Carrying the
        # whole conversation invites the model to answer from what it said
        # earlier rather than from the passages in front of it.
        q, a = history[-1]
        turns = f"\nEarlier in this conversation:\nQ: {q}\nA: {a[:400]}\n"

    scope = ""
    # A question about the author can trip the survey words — "what is he
    # working on recently" — and the clause below would then tell the model to
    # caveat a biography as an incomplete survey of a field. The site is not a
    # field, so the clause simply does not apply.
    if asks_for_a_survey(question) and not asks_about_the_author(question):
        # Two versions, because the single one went stale the day live search
        # landed. It told the model its evidence was "a fixed set of indexed
        # papers" even when half of it was abstracts fetched from arXiv,
        # PubMed and OpenAlex minutes earlier — instructing the model to
        # discount the very evidence that had been added to answer this kind
        # of question. Asked what is current in long-context language models
        # the model refused, with three recent abstracts in front of it.
        recent = [r for r in evidence if is_live(r.chunk.chunk_id)]
        if recent:
            scope = (
                f"\nThis question asks what a field is doing now. {len(recent)} of "
                "the passages above are abstracts of recent papers, fetched "
                "just now from literature search; the rest are from a fixed "
                "indexed corpus. Say what those recent papers are working on, "
                "cite them, and note that abstracts show what a paper claims "
                "rather than what it measured. Do not present a handful of "
                "papers as a complete survey of the field.\n"
            )
        else:
            scope = (
                "\nThis question asks about a field, not about these passages. "
                "Answer only what these papers show, and say in one clause that "
                "this is a fixed set of indexed papers rather than a survey of "
                "current literature.\n"
            )

    # Limitations are a question shape, not a topic, and the instruction only
    # applies when one is asked for — added here rather than to SYSTEM because
    # a rule in SYSTEM is paid for by every other question. Measured earlier in
    # this file's history: nine lines added to SYSTEM stopped a 3B model citing
    # at all, and the answer was refused for being ungrounded.
    if asks_for_limitations(question):
        scope += (
            "\nThis question asks what is wrong with the work, so answer it in "
            "the authors' own terms: report the limitations, caveats and future "
            "work that the passages themselves state, and attribute each to the "
            "paper it came from. Do not add weaknesses of your own, however "
            "reasonable — an unstated limitation is your opinion, not a finding. "
            "If a passage concedes nothing, say that the passages do not state "
            "the limitations rather than supplying some.\n"
        )

    user = f"""Passages:

{context}
{turns}{scope}
Question: {question}

Answer using only the passages above, citing each claim by number."""
    return SYSTEM, user
