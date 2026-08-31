"""Turning the model's [n] markers into checked citations.

The load-bearing idea: a citation the model reports is a claim about its own
reasoning, and a citation checked against the passages is a statement about the
output. Only the second kind is worth showing a reader.

So markers are resolved, not trusted. A marker pointing outside the evidence
list is deleted from the text — an invented [7] never reaches the page. What
survives is renumbered densely, so the reader sees [1][2][3] rather than the
gaps left by removing a fabrication, which would otherwise read as a bug.
"""

from __future__ import annotations

import re

from researchlens.types import Citation, Retrieved

#: One marker, or several inside one bracket. The grouped form is not a
#: nicety: a stronger model writes "[1, 2, 3, 4, 5, 6, 7]" naturally, and a
#: pattern that only matched "[n]" left that whole span as literal text
#: pointing at nothing while six of the seven sources vanished from the
#: evidence panel. A marker that resolves to no passage is the exact failure
#: this module exists to prevent, so it has to be parsed before it can be
#: checked.
_MARKER = re.compile(r"\[\s*(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\s*\]")

#: The numbers inside one bracket.
_NUMBERS = re.compile(r"\d{1,2}")

#: How much of a passage to show as the quote. Long enough to judge whether it
#: supports the claim, short enough that the evidence panel stays readable.
_QUOTE_CHARS = 320


def resolve(text: str, evidence: list[Retrieved]) -> tuple[str, list[Citation]]:
    """Strip unsupported markers, renumber the rest, and build the citations.

    Returns the rewritten answer and the citations it actually carries, in the
    order they first appear — so [1] is the first source a reader meets rather
    than whatever retrieval happened to rank first.
    """
    order: list[int] = []
    for m in _MARKER.finditer(text):
        for raw in _NUMBERS.findall(m.group(1)):
            n = int(raw)
            if 1 <= n <= len(evidence) and n not in order:
                order.append(n)

    renumber = {old: i + 1 for i, old in enumerate(order)}

    def rewrite(m: re.Match[str]) -> str:
        # A number outside the evidence list is a fabrication; remove it rather
        # than leave one pointing at nothing. A group can be partly invented,
        # so each is judged on its own and the bracket disappears only if
        # nothing in it survived.
        kept = [
            renumber[n]
            for n in (int(x) for x in _NUMBERS.findall(m.group(1)))
            if n in renumber
        ]
        return "".join(f"[{k}]" for k in kept)

    cleaned = _MARKER.sub(rewrite, text)
    # Removing a marker can leave " ." or a double space behind.
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    citations = []
    for old in order:
        r = evidence[old - 1]
        c = r.chunk
        quote = " ".join(c.text.split())
        if len(quote) > _QUOTE_CHARS:
            quote = quote[:_QUOTE_CHARS].rsplit(" ", 1)[0] + "…"
        citations.append(
            Citation(
                marker=renumber[old],
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                doc_title=c.doc_title,
                section_heading=c.section_heading,
                pages=c.pages,
                quote=quote,
            )
        )
    return cleaned, citations


def is_grounded(text: str, citations: list[Citation]) -> bool:
    """Whether an answer may be shown.

    An answer with no surviving citation is not shown as an answer. Either the
    model wrote from its own memory, or every marker it produced was invented —
    and both cases look identical to a reader, which is exactly why neither is
    allowed through.

    An explicit refusal is exempt: saying "the passages do not cover this" is
    the correct output and carries nothing to cite.
    """
    if citations:
        return True
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "could not find",
            "do not contain",
            "does not contain",
            "no evidence",
            "not discussed",
            "cannot answer",
            "insufficient evidence",
        )
    )
