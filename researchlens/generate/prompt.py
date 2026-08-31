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

from researchlens.types import Retrieved

SYSTEM = """You answer questions about scientific papers using only the numbered passages provided.

Rules:
- Use only the passages. If they do not answer the question, say so plainly.
- Cite every claim with the passage number in square brackets, like [2].
- A sentence stating a fact from the passages must carry a citation.
- Do not add background knowledge, even if you are confident it is correct.
- Quote exact numbers, dataset names and model names as they appear.
- Be concise. Three or four sentences is usually enough."""

NO_EVIDENCE = (
    "I could not find sufficient evidence in the indexed papers to answer that."
)


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
        block = f"[{i}] {c.doc_title} — {c.section_heading}, p{c.pages}\n{c.text}\n"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


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

    user = f"""Passages:

{context}
{turns}
Question: {question}

Answer using only the passages above, citing each claim by number."""
    return SYSTEM, user
