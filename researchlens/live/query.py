"""One place that decides what a question is *about*.

Three copies of this existed — one inline in arxiv.build_query, one `_STOP` in
pubmed, one in openalex — and they had drifted, which is how "recent research
scope in LLM" came to search all three indexes for the literal word *scope*.
arXiv returned "SCOPE: A Generative Approach for LLM Prompt Compression" and
"Quantifier Scope Interpretation in Language Learners and LLMs"; PubMed
returned "Global burden of 292 causes of death in 204 countries", which
reranked at -9.48. The reader had asked what the scope of recent LLM research
is, which is a shape of question, not a topic.

Adding "scope" to three separate sets would have fixed that question and left
the next one broken. So the sets are now one set, and a source that wants to
know the content words asks here.

Two jobs beyond stripping:

**Acronyms.** "llm" as a lexical term misses every paper that writes "large
language models" in full, which is most of the good ones. Every source here
supports OR, so an acronym becomes a group rather than a replacement — the
short form still matches, and the long form is added rather than substituted.

**Shape.** Whether a question asks for a survey, for limitations, or for a
plain fact changes which passages answer it. That judgement belongs with the
vocabulary it is made from, not scattered across the callers.
"""

from __future__ import annotations

import re

#: Question scaffolding. Words that say what *kind* of answer is wanted rather
#: than what it should be about, and which match nothing useful in an index.
#:
#: The union of the three lists this replaces, plus the ones that were missing
#: from all of them. Kept as one set on purpose: the failure being fixed here
#: was three sets disagreeing, and three sets cannot disagree if there is one.
STOP = {
    # interrogatives and glue
    "what", "which", "how", "why", "who", "when", "where", "is", "are", "was",
    "were", "does", "do", "did", "can", "could", "should", "would", "will",
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "with",
    "from", "by", "at", "as", "that", "this", "these", "those", "there",
    "about", "into", "over", "under", "between", "any", "some", "its", "it",
    # asking for a survey
    "current", "currently", "recent", "recently", "trend", "trends",
    "trending", "latest", "emerging", "nowadays", "lately", "modern", "new",
    "state", "art", "today",
    # asking for a shape of answer rather than a subject
    "scope", "scopes", "scoping", "overview", "landscape", "summary",
    "summarise", "summarize", "review", "reviews", "survey", "surveys",
    "area", "areas", "topic", "topics", "direction", "directions", "avenue",
    "avenues", "opportunity", "opportunities", "outlook", "progress",
    "advance", "advances", "advancement", "advancements", "development",
    "developments", "insight", "insights", "perspective", "perspectives",
    # asking about the literature rather than about a subject
    "research", "researches", "study", "studies", "paper", "papers", "work",
    "works", "literature", "publication", "publications", "article",
    "articles", "journal", "journals", "field", "fields", "domain",
    "problem", "problems", "gap", "gaps", "challenge", "challenges",
    "question", "questions", "open", "unresolved", "major", "main", "most",
    "important", "key", "good", "best", "common", "existing",
    # addressing the system
    "tell", "give", "show", "explain", "describe", "list", "find", "me",
    "please", "you", "your", "i", "my", "we", "our", "us",
    # verbs that carry no topic
    "used", "use", "uses", "using", "used", "make", "made", "get", "got",
    "have", "has", "had", "been", "being", "be",
}

#: Acronyms worth expanding, and what they expand to.
#:
#: Only ones where the short form is genuinely ambiguous or genuinely missing
#: from the prose of good papers. Deliberately excludes the tempting ones that
#: collide: "dr" is diabetic retinopathy here and "doctor" everywhere else,
#: "ml" is machine learning and also millilitres, and expanding either would
#: cost more than it bought.
ACRONYMS = {
    "llm": "large language model",
    "llms": "large language model",
    "vlm": "vision language model",
    "vlms": "vision language model",
    "rag": "retrieval augmented generation",
    "gnn": "graph neural network",
    "gnns": "graph neural network",
    "cnn": "convolutional neural network",
    "cnns": "convolutional neural network",
    "rnn": "recurrent neural network",
    "vit": "vision transformer",
    "vits": "vision transformer",
    "nlp": "natural language processing",
    "rl": "reinforcement learning",
    "ssl": "self supervised learning",
    "moe": "mixture of experts",
    "nas": "neural architecture search",
    "gan": "generative adversarial network",
    "gans": "generative adversarial network",
    "ocr": "optical character recognition",
    "asr": "automatic speech recognition",
    "fl": "federated learning",
}

_WORD = re.compile(r"[A-Za-z][A-Za-z-]*")


def terms(question: str, max_terms: int = 6, min_len: int = 2) -> list[str]:
    """The content words of a question, in order, deduplicated.

    `min_len` is 2 rather than 3 because requiring three characters silently
    dropped "AI" from "the major open problems in AI for radiology", leaving
    "radiology" alone — which, sorted by date, returned tantalum implants. A
    dropped acronym is not a small loss when the acronym is the subject.
    """
    seen: list[str] = []
    for w in _WORD.findall(question.lower()):
        if w in STOP or len(w) < min_len or w in seen:
            continue
        seen.append(w)
    return seen[:max_terms]


def expand(term: str) -> list[str]:
    """A term and its long form, or just the term.

    Returned as a list so a caller can join it with whatever OR its index
    speaks. Never replaces: an index that only has the acronym still matches.
    """
    long = ACRONYMS.get(term)
    return [term, long] if long else [term]


# --------------------------------------------------------------------------
# What shape of answer is being asked for
# --------------------------------------------------------------------------

#: Asking what a paper's own authors said was wrong, missing or unproven.
#:
#: "Limitation" is the obvious word and the rarest one in practice. Readers ask
#: for weaknesses, caveats, drawbacks, what a method cannot do, or what the
#: authors admit — all of which want the same passages.
_LIMITATION_WORDS = (
    "limitation", "limitations", "limited by", "weakness", "weaknesses",
    "shortcoming", "shortcomings", "drawback", "drawbacks", "caveat",
    "caveats", "downside", "downsides", "criticism", "fail", "fails",
    "failure", "failures", "cannot", "can't", "does not work", "doesn't work",
    "threats to validity", "future work", "unaddressed", "not addressed",
    "what is wrong", "problems with", "issues with", "critique",
)


def asks_for_limitations(question: str) -> bool:
    """Whether the reader wants what the authors themselves conceded.

    A separate question from "is this method good", and answerable in a way
    that opinion is not: a paper's limitations section is the authors on the
    record about their own work, and quoting it is reporting rather than
    judging. It is also the part a reader is least likely to reach on their
    own, being at the end of a paper nobody finishes.
    """
    q = f" {question.lower().strip()} "
    return any(w in q for w in _LIMITATION_WORDS)
