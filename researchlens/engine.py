"""The system, assembled.

One object that owns the library, the index and the providers, so the API is a
thin layer over it and the CLI can use exactly the same path. Anything the API
does that the engine cannot is a second implementation waiting to diverge.

Built once at startup and shared: the ONNX weights and the chunk index are tens
of seconds and hundreds of megabytes, and rebuilding them per request would
make the demo unusable and the memory ceiling unreachable.
"""

from __future__ import annotations

import sys
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
# Live search is triggered by the *shape of the question*, not by a guess at
# how well the corpus covers it.
#
# The first version predicted coverage from retrieval scores and it does not
# work. Measured over five questions with known answers:
#
#   covered  "does DL beat linear baselines"       3 papers   max cosine 0.866
#   thin     "current trends in long-context LLMs" 3 papers              0.845
#   absent   "current trends in computational pathology"  5 papers       0.740
#   absent   "gaps in automated renal CT reporting"       2 papers       0.829
#
# Neither the paper count nor the similarity separates them: a question the
# corpus cannot answer scores 0.829, above questions it answers well. bge-small
# returns high cosine for topically adjacent text whether or not the answer is
# actually present, which is the property that makes it a good retriever and a
# useless coverage detector.
#
# The question's own wording is the reliable signal. "What is current" is about
# a field, not about these papers, and is worth two seconds of live search
# regardless of what the corpus happens to hold — the prompt labels each source
# correctly, so combining them is safe.

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
        #: Why the last live search failed, if it did. Surfaced by /health so a
        #: broken live path is visible rather than merely quiet.
        self.last_live_error: str | None = None
        self._bundle_matrix = None

    # ---- startup ---------------------------------------------------------

    def load(self) -> None:
        """Load the corpus and build the index. Called once, before serving.

        A prebuilt bundle is used when present. That is what a deployed
        container gets: the parser derives each document id from the PDF's
        bytes, so parsing at startup would mean shipping ~600 MB of journal
        PDFs in the image and answering a redistribution question this project
        does not need to answer. Everything that serves a query is downstream
        of parsing, and that is ~24 MB.
        """
        data = Path(self.settings.data_dir)
        bundle = data / "bundle"

        if (bundle / "chunks.jsonl").exists():
            self._load_bundle(bundle)
        else:
            self.documents, self.skipped = load_library(data / "pdfs", data / "index")
            if not self.documents:
                raise RuntimeError(f"no readable papers under {data / 'pdfs'}")
            self.chunks = chunk_corpus(self.documents)

        self._by_id = {c.chunk_id: c for c in self.chunks}

        dense = DenseRetriever(model=self.settings.embedding_model)
        if (matrix := getattr(self, "_bundle_matrix", None)) is not None:
            # Hand the bundle's vectors straight in rather than re-embedding
            # 9,540 passages at every container start.
            dense._ids = [c.chunk_id for c in self.chunks]
            dense._matrix = matrix

        self.pipeline = RetrievalPipeline(
            dense=dense,
            bm25=BM25Retriever(),
            reranker=CrossEncoderReranker(model=self.settings.reranker_model),
        )
        if matrix is None:
            self.pipeline.index(self.chunks)
        else:
            self.pipeline._by_id = {c.chunk_id: c for c in self.chunks}
            self.pipeline.bm25.index(self.chunks)

    def _load_bundle(self, bundle: Path) -> None:
        """Read passages and vectors exported by scripts/export_bundle.py."""
        import json

        import numpy as np

        manifest = json.loads((bundle / "manifest.json").read_text())
        if manifest.get("embedding_model") != self.settings.embedding_model:
            # Vectors from one model scored against queries embedded by another
            # produce plausible nonsense, and nothing downstream would notice.
            raise RuntimeError(
                f"bundle was built with {manifest.get('embedding_model')!r} but this "
                f"instance uses {self.settings.embedding_model!r}"
            )

        self.chunks = []
        for line in (bundle / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                self.chunks.append(Chunk(**json.loads(line)))

        with np.load(bundle / "vectors.npz") as z:
            matrix = z["matrix"]
        if matrix.shape[0] != len(self.chunks):
            raise RuntimeError(
                f"bundle is inconsistent: {len(self.chunks)} passages but "
                f"{matrix.shape[0]} vectors — row order is the only thing linking "
                "them, so a mismatch would retrieve the wrong passage confidently"
            )
        self._bundle_matrix = matrix

        # The library listing is served from the bundle too, so /library keeps
        # working without the PDFs it was derived from.
        seen: dict[str, str] = {}
        for c in self.chunks:
            seen.setdefault(c.doc_id, c.doc_title)
        self.documents = [
            Document(doc_id=d, title=t, authors=[], sections=[], n_pages=0,
                     source_path="(bundled)")
            for d, t in seen.items()
        ]
        self.skipped = []

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
        except Exception as e:
            # Live search is an enhancement, not a dependency: a network
            # failure degrades to a corpus-only answer rather than an error,
            # because the local deployment is meant to work offline.
            #
            # But it says so. An earlier version returned [] silently, and when
            # live search stopped working the only symptom was an answer with
            # no recent evidence in it — indistinguishable from a corpus that
            # simply had nothing to add. Swallowing the reason is the exact
            # failure mode this project exists to avoid.
            self.last_live_error = f"{type(e).__name__}: {e}"
            print(f"  live search unavailable — {self.last_live_error}", file=sys.stderr)
            return []
        self.last_live_error = None
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

    def _merge_live(
        self, question: str, fetched: list[Retrieved], corpus: list[Retrieved]
    ) -> list[Retrieved]:
        """Combine live and corpus evidence, keeping only what is relevant.

        Live results cannot simply be prepended. Sources are queried in
        parallel and one of them is often wrong for the question: asked about
        long-context language models, PubMed returns "Sequence modeling from
        molecular to genome scale", because that is the nearest thing a
        biomedical index holds. Half the evidence then being off-topic, the
        model correctly answered that it could not tell — a regression caused
        by adding a source, not by the source being bad.

        So live abstracts compete on relevance like everything else, through
        the same cross-encoder that ranks corpus passages. An abstract that
        does not answer the question drops out, whichever index it came from,
        and nothing needs to know which sources suit which questions.
        """
        pool = fetched + corpus
        if self.pipeline is None or self.pipeline.reranker is None:
            return pool[: SERVING.top_k]

        by_id = {r.chunk.chunk_id: r for r in pool}
        ranked = self.pipeline.reranker.rerank(
            question, [r.chunk for r in pool], SERVING.top_k
        )
        return [
            Retrieved(
                chunk=by_id[cid].chunk,
                score=score,
                dense_score=by_id[cid].dense_score,
                bm25_score=by_id[cid].bm25_score,
                rerank_score=score,
                sources=by_id[cid].sources,
            )
            for cid, score in ranked
        ]

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
            live = asks_for_a_survey(question)

        if live:
            fetched = await self.live_evidence(question)
            if fetched:
                evidence = self._merge_live(question, fetched, evidence)

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
