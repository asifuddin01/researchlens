"""The system, assembled.

One object that owns the library, the index and the providers, so the API is a
thin layer over it and the CLI can use exactly the same path. Anything the API
does that the engine cannot is a second implementation waiting to diverge.

Built once at startup and shared: the ONNX weights and the chunk index are tens
of seconds and hundreds of megabytes, and rebuilding them per request would
make the demo unusable and the memory ceiling unreachable.
"""

from __future__ import annotations

import time
from pathlib import Path

from researchlens.config import Settings
from researchlens.generate.citations import is_grounded, resolve
from researchlens.generate.prompt import NO_EVIDENCE, asks_for_a_survey
from researchlens.generate.provider import GenerationRequest
from researchlens.ingest.chunk import chunk_corpus
from researchlens.ingest.library import load_library
from researchlens.retrieval.bm25 import BM25Retriever
from researchlens.retrieval.dense import DenseRetriever
from researchlens.retrieval.pipeline import RetrievalConfig, RetrievalPipeline
from researchlens.retrieval.rerank import CrossEncoderReranker
from researchlens.types import Answer, Chunk, Document, Retrieved

#: What the API serves. The ablation may yet show the reranker earns nothing on
#: this corpus; if so this becomes `hybrid (RRF)` and the change is one line.
#: Distinct papers below which a survey question is answered from live search
#: as well as the corpus. Four, matching the survey floor in the benchmark
#: builder: one paper is an anecdote, and a "current trends" answer drawn from
#: three papers is a reading list rather than a trend.
LIVE_SUPPORT_FLOOR = 4

SERVING = RetrievalConfig(
    label="hybrid + rerank", use_dense=True, use_bm25=True, use_rerank=True,
    candidates=30, top_k=8,
)


class Engine:
    """Library, index and providers, ready to answer."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self.pipeline: RetrievalPipeline | None = None
        self.skipped: list[str] = []

    # ---- startup ---------------------------------------------------------

    def load(self) -> None:
        """Parse, chunk and index. Called once, before serving."""
        data = Path(self.settings.data_dir)
        self.documents, self.skipped = load_library(data / "pdfs", data / "index")
        if not self.documents:
            raise RuntimeError(f"no readable papers under {data / 'pdfs'}")

        self.chunks = chunk_corpus(self.documents)
        self._by_id = {c.chunk_id: c for c in self.chunks}

        self.pipeline = RetrievalPipeline(
            dense=DenseRetriever(model=self.settings.embedding_model),
            bm25=BM25Retriever(),
            reranker=CrossEncoderReranker(model=self.settings.reranker_model),
        )
        self.pipeline.index(self.chunks)

    @property
    def ready(self) -> bool:
        return self.pipeline is not None and bool(self.chunks)

    # ---- providers -------------------------------------------------------

    def provider(self, name: str):
        """Return a provider by name, or raise with what is actually available.

        Falling back silently to the other provider would make the two-model
        comparison meaningless — a visitor would be told they were reading the
        local model's answer when they were not.
        """
        if name == "hosted":
            if not self.settings.hosted_available:
                raise ValueError(
                    "the hosted provider is not configured; this instance serves "
                    "the local model only"
                )
            from researchlens.generate.openai_compat import OpenAICompatProvider

            return OpenAICompatProvider(
                base_url=self.settings.hosted_base_url or "",
                api_key=self.settings.hosted_api_key,
                model=self.settings.hosted_model,
            )
        if name == "local":
            from researchlens.generate.ollama import OllamaProvider

            return OllamaProvider(
                host=self.settings.ollama_host, model=self.settings.ollama_model
            )
        raise ValueError(f"unknown provider {name!r}; expected 'local' or 'hosted'")

    # ---- answering -------------------------------------------------------

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[list[Retrieved], float]:
        if self.pipeline is None:
            raise RuntimeError("Engine.load() has not been called")
        config = SERVING if top_k is None else RetrievalConfig(
            label=SERVING.label, use_dense=True, use_bm25=True, use_rerank=True,
            candidates=SERVING.candidates, top_k=top_k,
        )
        return self.pipeline.timed_search(question, config)

    async def live_evidence(self, question: str, max_results: int = 6) -> list[Retrieved]:
        """Fetch recent papers when the corpus cannot speak to the question.

        Returned as `Retrieved` so the rest of the pipeline is unchanged; the
        chunks carry an `arxiv:` id and an "abstract" page, so a reader is
        never shown an abstract as though it were a passage from a full paper.
        """
        from researchlens.live import arxiv, search as live_search

        try:
            papers = await live_search.search(question, per_source=max_results // 2 or 3)
        except Exception:
            # Live search is an enhancement, not a dependency. A network
            # failure must degrade to a corpus-only answer, never to an error:
            # the local deployment is supposed to work with no network at all.
            return []
        return [Retrieved(chunk=c, score=0.0, sources=frozenset({"live"}))
                for c in arxiv.to_chunks(papers)]

    def corpus_support(self, question: str, threshold: float = 0.62) -> int:
        """How many distinct papers the corpus offers on this question.

        Used to decide whether live search is warranted. Deliberately a count
        of *papers*, not passages: eight passages from one paper is one paper's
        worth of evidence, and answering "what is the field doing" from it
        would be the same failure in a new costume.
        """
        hits, _ms = self.retrieve(question, top_k=20)
        return len({r.chunk.doc_id for r in hits if r.score >= threshold})

    async def ask(
        self,
        question: str,
        provider_name: str = "local",
        history=None,
        live: bool | None = None,
    ) -> Answer:
        """Retrieve, generate, then check what came back.

        The order matters. Retrieval happens first and, when it returns
        nothing, the model is never called — a model asked a question with no
        context answers from its training, fluently, and a fluent unsourced
        answer is worse than a refusal because a reader cannot tell them apart.
        """
        evidence, retrieval_ms = self.retrieve(question)
        provider = self.provider(provider_name)

        # Whether to reach outside the corpus is decided from what the corpus
        # actually holds, not from a flag someone has to remember to set. A
        # question asking what a field is doing *now*, answered from a fixed
        # corpus, is the failure that looks most like success — real citations,
        # eight-year-old evidence, no indication of either.
        if live is None:
            live = asks_for_a_survey(question) and len(
                {r.chunk.doc_id for r in evidence}
            ) < LIVE_SUPPORT_FLOOR

        if live:
            fetched = await self.live_evidence(question)
            # Live evidence goes first: for a question about what is recent,
            # a 2026 abstract is better evidence than a 2017 passage, and the
            # model cites the passages it reads first more reliably.
            evidence = fetched + evidence[: max(0, SERVING.top_k - len(fetched))]

        if not evidence:
            return Answer(
                question=question, text=NO_EVIDENCE, citations=[],
                model=provider.model, provider=provider.name,
                retrieval_ms=retrieval_ms, generation_ms=0.0,
            )

        started = time.perf_counter()
        raw = await provider.generate(
            GenerationRequest(question=question, evidence=evidence, history=history)
        )
        generation_ms = (time.perf_counter() - started) * 1000.0

        text, citations = resolve(raw, evidence)
        if not is_grounded(text, citations):
            # Every marker was invented, or none was produced. Either way the
            # answer is unsupported, and showing it would be the exact failure
            # this project exists to prevent.
            text, citations = NO_EVIDENCE, []

        return Answer(
            question=question, text=text, citations=citations,
            model=provider.model, provider=provider.name,
            retrieval_ms=retrieval_ms, generation_ms=generation_ms,
        )
