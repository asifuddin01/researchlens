"""The system, assembled.

One object that owns the library, the index and the providers, so the API is a
thin layer over it and the CLI can use exactly the same path. Anything the API
does that the engine cannot is a second implementation waiting to diverge.

Built once at startup and shared: the ONNX weights and the chunk index are tens
of seconds and hundreds of megabytes, and rebuilding them per request would
make the demo unusable and the memory ceiling unreachable.
"""

from __future__ import annotations

import os
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
from researchlens.uploads import UploadStore

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

#: Candidates reranked per query. The cross-encoder is 95% of retrieval time —
#: 45 ms without it locally, 855 ms with — so this is the one number that
#: decides whether a demo feels responsive.
#:
#: Thirty is right where CPU is plentiful. On the free Space it is not: no
#: CoreML, a weaker core, and the same thirty passes took 13.6 s against 855 ms
#: locally. Lowering it there trades a little ordering quality for a demo that
#: answers while someone is still watching, and the ablation has not yet shown
#: the reranker earns those extra candidates anyway.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))

SERVING = RetrievalConfig(
    label="hybrid + rerank", use_dense=True, use_bm25=True, use_rerank=True,
    candidates=RERANK_CANDIDATES, top_k=8,
)


class Engine:
    """Library, index and providers, ready to answer."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self.pipeline: RetrievalPipeline | None = None
        #: Papers readers add at run time, held per session and never merged
        #: into the corpus. See researchlens/uploads.py for why.
        self.uploads: UploadStore | None = None
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
        self.uploads = UploadStore(
            embedding_model=self.settings.embedding_model,
            # The corpus reranker, borrowed. Sharing it is what keeps an
            # uploaded passage's score comparable with a corpus passage's.
            reranker=self.pipeline.reranker,
            corpus_doc_ids={d.doc_id for d in self.documents},
        )

        if matrix is None:
            self.pipeline.index(self.chunks)
        else:
            # The bundle already holds the dense vectors, so only BM25 needs
            # building — but the pipeline's own bookkeeping must still happen,
            # or the row indices that restrict retrieval to chosen papers will
            # not exist. Reaching past index() to set one field was how that
            # broke.
            self.pipeline.index_without_dense(self.chunks)
            # A bundle carries its vectors, so nothing has touched the encoder
            # and it would first load on someone's question — half a second
            # charged to the first visitor after every cold start, and again to
            # the first upload. Pay it here, where nobody is waiting.
            self._warm_encoder(dense)

    @staticmethod
    def _warm_encoder(dense) -> None:
        try:
            # Consumed, not just called: fastembed returns a generator, and an
            # unconsumed one warms nothing at all.
            list(dense._lazy_encoder().embed(["warm"]))
        except Exception as e:  # noqa: BLE001
            # Not fatal — the first real call will load it. Worth saying,
            # because a failure here means the weights are missing and every
            # query is about to fail for a reason this line already knows.
            print(f"  could not warm the encoder: {e}", file=sys.stderr)

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

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        doc_ids: set[str] | None = None,
        session: str | None = None,
    ) -> tuple[list[Retrieved], float]:
        """Rank the corpus, and this session's uploaded papers alongside it.

        `session` names an upload session, not a user: uploaded papers live in
        their own index and are merged here, so every caller — the API, the CLI
        and the Space's GPU path — reads them without knowing they exist.
        """
        if self.pipeline is None:
            raise RuntimeError("Engine.load() has not been called")
        k = SERVING.top_k if top_k is None else top_k
        config = SERVING if top_k is None else RetrievalConfig(
            label=SERVING.label, use_dense=True, use_bm25=True, use_rerank=True,
            candidates=SERVING.candidates, top_k=top_k,
        )
        results, ms = self.pipeline.timed_search(question, config, doc_ids)

        if self.uploads is not None and session:
            started = time.perf_counter()
            mine = self.uploads.search(session, question, config, doc_ids)
            if mine:
                results = self._merge_uploaded(question, mine, results, k)
            ms += (time.perf_counter() - started) * 1000.0
        return results, ms

    def _merge_uploaded(
        self, question: str, uploaded: list[Retrieved], corpus: list[Retrieved], k: int
    ) -> list[Retrieved]:
        """Give a reader's own papers guaranteed room in the evidence.

        Not because they rank higher — often they do not; a hundred and one
        indexed papers contain a better-matching passage than one manuscript
        more often than not. But someone who uploads a paper and asks a
        question is asking about *that paper*, and an answer built entirely
        from the corpus would be a correct answer to a question they did not
        ask.

        Half the slots, at most. The other half stays with the corpus, because
        the thing worth having is the uploaded paper read *against* the
        literature — "how does this compare" is the question the corpus makes
        answerable, and reserving everything would take it away.
        """
        docs = len({r.chunk.doc_id for r in uploaded})
        if not corpus:
            # Nothing to interleave with: the reader selected only their own
            # papers, or the corpus had nothing to say. Reserving half would
            # then return four passages where eight were available — the same
            # failure the diversity cap caused when a reader chose one corpus
            # paper and got three passages and a refusal.
            slots, per_doc = min(len(uploaded), k), k
        else:
            slots = min(len(uploaded), max(2, k // 2))
            # One uploaded paper should be readable in depth; three should not
            # have one of them crowd out the other two.
            per_doc = max(2, -(-slots // max(docs, 1)))
        return self._merge_reserved(question, uploaded, corpus, slots, per_doc, 2, k)

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

    # ---- uploads ---------------------------------------------------------

    def add_upload(self, session: str, raw: bytes, filename: str) -> Document:
        """Index one uploaded PDF for this session. Raises `UploadError`."""
        if self.uploads is None:
            raise RuntimeError("Engine.load() has not been called")
        return self.uploads.add(session, raw, filename)

    def uploaded_documents(self, session: str | None) -> list[Document]:
        if self.uploads is None or not session:
            return []
        return self.uploads.documents(session)

    def uploaded_passage_counts(self, session: str | None) -> dict[str, int]:
        return {} if self.uploads is None else self.uploads.passage_counts(session)

    def drop_uploads(self, session: str) -> None:
        if self.uploads is not None:
            self.uploads.drop(session)

    def corpus_support(self, question: str, threshold: float = 0.62) -> int:
        """How many distinct papers the corpus offers on this question.

        Used to decide whether live search is warranted. Deliberately a count
        of *papers*, not passages: eight passages from one paper is one paper's
        worth of evidence, and answering "what is the field doing" from it
        would be the same failure in a new costume.
        """
        hits, _ms = self.retrieve(question, top_k=20)
        return len({r.chunk.doc_id for r in hits if r.score >= threshold})

    @staticmethod
    def _diversify(results: list[Retrieved], per_doc: int, limit: int) -> list[Retrieved]:
        """Cap how many passages any one paper contributes.

        Without this, asked what the current research trend in LLMs is, six of
        eight passages came from a single survey paper. The answer was then a
        summary of that paper wearing the question's clothes — and a reader
        cannot tell a synthesis across the literature from a paraphrase of one
        document, because both arrive with citations.
        """
        kept: list[Retrieved] = []
        seen: dict[str, int] = {}
        for r in results:
            n = seen.get(r.chunk.doc_id, 0)
            if n >= per_doc:
                continue
            seen[r.chunk.doc_id] = n + 1
            kept.append(r)
            if len(kept) >= limit:
                break
        return kept

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
        # Live and corpus evidence are ranked separately, then interleaved,
        # rather than competing in one pool.
        #
        # Ranking them together does not work for the question live search
        # exists to answer. Asked what the current research trend in LLMs is,
        # six recent papers were fetched and one survived: a cross-encoder
        # scores topical relevance, and a 2024 survey already in the corpus
        # reads as more relevant to "research trend in LLMs" than a 2026
        # abstract about one narrow result. It is more relevant. It is also
        # eight years of literature out of date, which is the thing being
        # asked about.
        #
        # So recency gets guaranteed room: at least half the evidence on a
        # survey question comes from the live half, if the live half found
        # anything at all.
        slots = min(len(fetched), max(2, SERVING.top_k // 2))
        # One live abstract per paper: six abstracts from six papers is a view
        # of the field, which is what a survey question asked for.
        return self._merge_reserved(question, fetched, corpus, slots, 1, 2, SERVING.top_k)

    def _merge_reserved(
        self,
        question: str,
        reserved: list[Retrieved],
        rest: list[Retrieved],
        slots: int,
        reserved_per_doc: int,
        rest_per_doc: int,
        limit: int,
    ) -> list[Retrieved]:
        """Rank two pools separately, then give one of them guaranteed room.

        Shared by live search and uploads because they are the same problem in
        two costumes: evidence the reader has a reason to want, which the
        cross-encoder would otherwise rank below a corpus passage that is
        merely more on-topic. Two copies of this would have drifted the first
        time one of them was tuned.

        Cross-encoder scores are absolute rather than corpus-relative, so a
        single ranking would in fact be *sound* — the reason for two is
        editorial, not numerical, and it is written out at each call site.
        """
        if self.pipeline is None or self.pipeline.reranker is None:
            return (reserved + rest)[:limit]

        def rerank(pool: list[Retrieved], k: int) -> list[Retrieved]:
            if not pool or k <= 0:
                return []
            by_id = {r.chunk.chunk_id: r for r in pool}
            return [
                Retrieved(
                    chunk=by_id[cid].chunk, score=score,
                    dense_score=by_id[cid].dense_score,
                    bm25_score=by_id[cid].bm25_score,
                    rerank_score=score, sources=by_id[cid].sources,
                )
                for cid, score in self.pipeline.reranker.rerank(
                    question, [r.chunk for r in pool], k
                )
            ]

        head = self._diversify(rerank(reserved, slots * 2), reserved_per_doc, slots)
        tail = self._diversify(rerank(rest, limit), rest_per_doc, limit - len(head))
        return head + tail

    async def ask(
        self,
        question: str,
        provider_name: str = "local",
        history=None,
        live: bool | None = None,
        doc_ids: set[str] | None = None,
        session: str | None = None,
    ) -> Answer:
        """Retrieve, generate, then check what came back.

        The order matters. Retrieval happens first and, when it returns
        nothing, the model is never called — a model asked a question with no
        context answers from its training, fluently, and a fluent unsourced
        answer is worse than a refusal because a reader cannot tell them apart.
        """
        evidence, retrieval_ms = self.retrieve(question, doc_ids=doc_ids, session=session)
        provider = self.provider(provider_name)

        # Whether to reach outside the corpus is decided from what the corpus
        # actually holds, not from a flag someone has to remember to set. A
        # question asking what a field is doing *now*, answered from a fixed
        # corpus, is the failure that looks most like success — real citations,
        # eight-year-old evidence, no indication of either.
        if live is None:
            # Never when a subset is chosen. Asking "what does this paper say"
            # and receiving abstracts from elsewhere answers a question nobody
            # asked, and a reader who selected two papers has said plainly that
            # the rest of the literature is not what they want.
            live = (
                asks_for_a_survey(question)
                and not doc_ids
                and not self.uploaded_documents(session)
            )

        if live:
            fetched = await self.live_evidence(question)
            if fetched:
                evidence = self._merge_live(question, fetched, evidence)

        # One paper should not fill the context on an ordinary question. When a
        # reader has *chosen* the papers, the cap is the opposite of what they
        # asked for: selecting a single paper and receiving three passages from
        # it is a worse answer than the corpus could give, and the refusal that
        # followed looked like the paper had nothing to say.
        # _merge_uploaded has already given the reader's own papers their
        # slots; capping again here would take them straight back.
        if not (doc_ids or self.uploaded_documents(session)):
            evidence = self._diversify(evidence, per_doc=3, limit=SERVING.top_k)
        else:
            evidence = evidence[: SERVING.top_k]

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
