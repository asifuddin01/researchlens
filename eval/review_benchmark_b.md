# Benchmark B — coverage review

Each question with its current label and what the system actually did.
Change `expect` in `eval/benchmark_b.jsonl` where the label is wrong.

Disagreements first — these are where the label and the behaviour differ,
and where a wrong label is doing the most damage to the score.


---

### B012 — **DISAGREES**

**Q:** What are the major open problems in vision foundation models?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 1 citations, 9s

> The major open problems in vision foundation models include the degree of visual content hallucination [1] and fine-grained understanding of visual content [1].

---

### B016 — **DISAGREES**

**Q:** What are the major unresolved problems in VLM grounding?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 2 citations, 15s

> The major unresolved problems in VLM grounding are the lack of clear specification of scale [1] and the need for comprehensive grounding rules [2].

---

### B023 — **DISAGREES**

**Q:** What are the current research directions in foundation models for biology?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 3 citations, 14s

> These passages show that current research directions in foundation models for biology include exploring the impact of larger perturbation datasets, comparing architectural differences between model families, and further exploring models' performance relative to various gene categories and relative expression levels of causal genes [1]. Ad…

---

### B030 — **DISAGREES**

**Q:** Which studies perform prediction → new experiment → validation?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 1 citations, 8s

> [1] performs prediction → new experiment → validation.

---

### B032 — **DISAGREES**

**Q:** What are the major gaps between in-silico biological prediction and experimental validation?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 1 citations, 10s

> The major gaps between in-silico biological prediction and experimental validation are not explicitly detailed in the provided passages. However, it can be inferred that there are discrepancies between the predictions made by in-silico models and the experimental outcomes. [1] discusses factors influencing the success of in-silico perturb…

---

### B045 — **DISAGREES**

**Q:** Which research directions are rapidly growing, based on recent publications?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 1 citations, 11s

> [1] Rapidly growing topic directions include Video-Based Health Interventions for Attitude Change and Social Media Engagement Strategies for Education.

---

### B046 — **DISAGREES**

**Q:** What are the most underexplored research gaps in a selected field?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 1 citations, 8s

> [1] Process supervision is currently underexplored.

---

### B047 — **DISAGREES**

**Q:** Which research gaps are repeatedly identified across recent papers?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 1 citations, 10s

> These passages show that recent papers repeatedly identify the challenges and consequences associated with deploying deep learning models in clinical environments as a research gap [1].

---

### B049 — **DISAGREES**

**Q:** What opportunities emerge when findings from two previously separate research areas are combined?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 1 citations, 9s

> When findings from two previously separate research areas are combined, opportunities emerge for large video-audio-language models, learning from extended feedback and trial-and-errors, understanding and generating codebases, adapting AI models to understand scientific data such as gene sequences, and developing strong reasoning from link…

---

### B050 — **DISAGREES**

**Q:** What promising research problems can be identified from recent literature that are not yet well solved?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 3 citations, 21s

> [1] proposes two potential research problems for future exploration in the field of diabetic retinopathy detection. [2] suggests that the current autoregressive mechanism for generating text may not be sufficient for a language model to be built toward a general problem solver, and identifies the need for alternative mechanisms. [3] highl…

---

### B052 — **DISAGREES**

**Q:** When do deep learning methods fail to outperform simpler baselines?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 1 citations, 8s

> Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines [1].

---

### B053 — **DISAGREES**

**Q:** What external quantitative references are commonly used to validate biological predictions?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 1 citations, 11s

> [1] further validated the epigenomic information with independent external references including Human Brain Atlas (HBA) snATAC cCREs and H3K27ac cCREs.

---

### B054 — **DISAGREES**

**Q:** Which recent studies use real biological measurements rather than only simulated data?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 1 citations, 11s

> [1] Recent amortized causal graph frameworks use real biological measurements rather than only simulated data.

---

### B056 — **DISAGREES**

**Q:** What specific diseases remain comparatively underrepresented in the literature?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 2 citations, 10s

> Underrepresented diseases in the literature include Alzheimer's disease [1], immune disorders [1], and those in underrepresented groups for ML model training and validation [2].

---

### B059 — **DISAGREES**

**Q:** How often are code, data, and pretrained models publicly released in a selected field?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 3 citations, 13s

> [1] pretrained models are publicly available [1], [2] code and data are publicly available [3], [2].

---

### B067 — **DISAGREES**

**Q:** Can current AI research assistants reliably identify research gaps? What evidence supports the answer?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 1 citations, 10s

> Current AI research assistants cannot reliably identify research gaps [1]. This is supported by the fact that existing benchmarks mainly evaluate visual understanding in isolation, leaving open whether models can arbitrate between visual evidence and user-provided language [1].

---

### B071 — **DISAGREES**

**Q:** Identify a research trend supported by multiple independent recent sources.

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 1 citations, 11s

> This set of papers shows that transformer language model capacity has increased from 100 million parameters to 17 billion parameters, with improvements in text synthesis and downstream NLP tasks. [1]

---

### B072 — **DISAGREES**

**Q:** Identify a claimed research trend that is not well supported by current evidence.

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 1 citations, 11s

> [1] Masked Autoencoders Are Scalable Vision Learners D V X — Partial Fine-tuning, p7-8 suggests that linear evaluation is not well correlated with transfer learning performance for object detection.

---

### B074 — **DISAGREES**

**Q:** Determine whether a recent method truly improves over the strongest contemporary baseline.

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 1 citations, 11s

> These passages show that DPO improves over the strongest contemporary baseline, Best of 128, in the Anthropic-HH dataset [1].

---

### B075 — **DISAGREES**

**Q:** Identify a recent research gap that could lead to a feasible thesis project.

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 3 citations, 10s

> This set of passages shows research gaps in privacy-preserving diabetic retinopathy detection using federated learning [1], improving the factuality of language model generations with respect to context and world knowledge [2], and characterizing the capabilities, shortcomings, and biases of visual models like CLIP [3]. These gaps could l…

---

### B076 — **DISAGREES**

**Q:** Propose a research opportunity supported by recent literature and explain the evidence behind it.

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 3 citations, 9s

> [1] Proposes a research opportunity in computationally efficient deep learning models for diabetic retinopathy detection. Evidence is provided by [2] highlighting the potential of retinal imaging for systemic disease surveillance and [3] detailing a systematic literature review on efficient models.

---

### B077 — **DISAGREES**

**Q:** Identify an underexplored combination of methods, datasets, or research areas.

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 1 citations, 13s

> [1] We have shown that process supervision can be used to train much more reliable reward models than outcome supervision in the domain of mathematical reasoning. We have also shown that active learning can be used to lower the cost of human data collection by surfacing only the most valuable model completions for human feedback. We relea…

---

### B078 — **DISAGREES**

**Q:** What experimental validation would most strongly test a current computational prediction?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 4 citations, 11s

> What experimental validation would most strongly test a current computational prediction? This is a fixed set of indexed papers rather than a survey of current literature showing various benchmarking and validation methods [1], [2], [3], [4].

---

### B079 — **DISAGREES**

**Q:** Based on current literature, what research questions are likely to remain important over the next few years?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **answer**, 3 citations, 13s

> Based on current literature, research questions related to the limitations of current pretraining objectives and the need for more sophisticated statistical methods remain important over the next few years [1], [2]. Additionally, the need for new experimental designs and models that jointly span multiple causal signatures is also likely t…

---

### B001 — agrees

**Q:** What are the major current research trends in large language models?

- label: `expect: scoped` (coverage: partial)
- topics matched: language models=31
- system: **refuse**, 0 citations, 64s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B002 — agrees

**Q:** What are the most important open problems in LLM reasoning?

- label: `expect: scoped` (coverage: partial)
- topics matched: language models=31, reasoning=3
- system: **answer**, 1 citations, 18s

> The most important open problems in LLM reasoning are improving the factuality of language model generations with respect to context and world knowledge [1]. Improving the factuality of language model generations can potentially improve multi-step reasoning abilities of language models [1].

---

### B003 — agrees

**Q:** What are the current research directions in efficient LLM training and inference?

- label: `expect: scoped` (coverage: partial)
- topics matched: language models=31, efficient training=7
- system: **answer**, 3 citations, 17s

> These passages show that current research directions in efficient LLM training and inference include exploring speculative-decoding modules in recent LLMs for efficient high-quality classification [1]. They also emphasize the need for specialized strategies to integrate retrieval with language generation models to manage contexts exceedin…

---

### B004 — agrees

**Q:** What are the major unresolved problems in retrieval-augmented generation?

- label: `expect: scoped` (coverage: partial)
- topics matched: retrieval-augmented generation=4
- system: **answer**, 1 citations, 10s

> The major unresolved problem in retrieval-augmented generation is how the model behaves if retrieval goes wrong [1].

---

### B005 — agrees

**Q:** What are the current trends in long-context language models?

- label: `expect: scoped` (coverage: partial)
- topics matched: language models=31, long context=0
- system: **answer**, 3 citations, 7s

> These passages show that current trends in long-context language models include the development of benchmarks for prompt injection attacks [1], the use of long-context models for genome prediction [2], and the introduction of methods for proactive context management [3]. However, they do not provide information on current trends as these …

---

### B006 — agrees

**Q:** What are the major research gaps in AI agents?

- label: `expect: scoped` (coverage: thin)
- topics matched: agents=2
- system: **refuse**, 0 citations, 9s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B007 — agrees

**Q:** What are the current research directions in multimodal LLMs?

- label: `expect: scoped` (coverage: partial)
- topics matched: language models=31, multimodal=4
- system: **answer**, 2 citations, 14s

> [1] [2] Current research directions in multimodal LLMs include addressing challenges such as the inclusion of irrelevant documents in retrieval, the development of specialized strategies to integrate retrieval with language generation models, and the need for standardized evaluation frameworks. Future research should also focus on multimo…

---

### B008 — agrees

**Q:** What are the main open problems in NLP evaluation?

- label: `expect: scoped` (coverage: thin)
- topics matched: evaluation=3
- system: **refuse**, 0 citations, 7s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B009 — agrees

**Q:** What are the current research trends in hallucination detection and mitigation?

- label: `expect: scoped` (coverage: thin)
- topics matched: hallucination=1
- system: **answer**, 6 citations, 14s

> These passages show that current research trends in hallucination detection and mitigation include the use of LLMs for detection and mitigation [1], the use of dynamic alignment compensation for inference-time hallucination mitigation [2], the use of cross-model diffusion for rumor detection [3], the transformation of LLMs from passive re…

---

### B010 — agrees

**Q:** What are the major gaps in factuality evaluation for generative models?

- label: `expect: scoped` (coverage: thin)
- topics matched: hallucination=1, evaluation=3
- system: **answer**, 1 citations, 12s

> The major gaps in factuality evaluation for generative models are that it is unclear how to measure honesty in purely generative models [1]. Additionally, measuring the harms of language models also poses many challenges [1].

---

### B011 — agrees

**Q:** What are the current research directions in computer vision?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **refuse**, 0 citations, 14s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B013 — agrees

**Q:** What are the current trends in image and video generation?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **refuse**, 0 citations, 14s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B014 — agrees

**Q:** What are the major research gaps in 3D vision?

- label: `expect: scoped` (coverage: thin)
- topics matched: 3d vision=1
- system: **refuse**, 0 citations, 12s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B015 — agrees

**Q:** What are the current research directions in vision-language models?

- label: `expect: scoped` (coverage: partial)
- topics matched: language models=31, multimodal=4
- system: **refuse**, 0 citations, 18s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B017 — agrees

**Q:** What are the current trends in multimodal reasoning?

- label: `expect: scoped` (coverage: partial)
- topics matched: reasoning=3, multimodal=4
- system: **refuse**, 0 citations, 15s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B018 — agrees

**Q:** What are the major gaps in medical vision-language models?

- label: `expect: scoped` (coverage: partial)
- topics matched: language models=31, multimodal=4, medical imaging=8
- system: **answer**, 8 citations, 16s

> The passages provided do not discuss medical vision-language models or gaps in such models. They focus on vision-language pre-training methods and models for general vision-language understanding and generation. Therefore, there are no major gaps in medical vision-language models to be discussed based on the given passages. [1], [2], [3],…

---

### B019 — agrees

**Q:** What are the current research directions in bioinformatics?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **refuse**, 0 citations, 16s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B020 — agrees

**Q:** What are the major open problems in single-cell genomics?

- label: `expect: scoped` (coverage: partial)
- topics matched: single-cell=14
- system: **answer**, 4 citations, 14s

> The major open problems in single-cell genomics include the mismatch between model complexity and the completeness of information available in the data [1]. Additionally, the application of single-cell perturbation sequencing at large scales faces substantial practical and economic barriers [2]. Lastly, deep-learning-based gene perturbati…

---

### B021 — agrees

**Q:** What are the current trends in gene perturbation effect prediction?

- label: `expect: scoped` (coverage: partial)
- topics matched: perturbation prediction=12
- system: **answer**, 7 citations, 14s

> What these papers show is a fixed set of indexed papers rather than a survey of current trends in gene perturbation effect prediction. [1], [2], [3], [4], [5], [6], [7] demonstrate various methods and their performance in predicting gene perturbation effects, but do not provide a comprehensive overview of current trends.

---

### B022 — agrees

**Q:** What are the major research gaps in gene regulatory network inference?

- label: `expect: scoped` (coverage: partial)
- topics matched: gene regulatory networks=6
- system: **answer**, 5 citations, 15s

> The major research gaps in gene regulatory network inference are: [1] Existing benchmarks are insufficiently controlled to evaluate causal methods for gene regulatory network inference from single-cell RNA-seq data. [2] Biological networks used for evaluation do not fully capture ground-truth GRNs, and reported connections are often biase…

---

### B024 — agrees

**Q:** What are the major limitations of current single-cell foundation models?

- label: `expect: scoped` (coverage: partial)
- topics matched: single-cell=14
- system: **answer**, 3 citations, 13s

> Current single-cell foundation models have limitations including suboptimal performance in predicting combined-perturbation effects [1], limited generalizability [1], and do not outperform simple linear baselines [2]. Additionally, they assume additive perturbation effects and do not explicitly model higher-order interactions or complex l…

---

### B025 — agrees

**Q:** What are the current approaches for predicting gene perturbation responses?

- label: `expect: scoped` (coverage: partial)
- topics matched: perturbation prediction=12
- system: **answer**, 4 citations, 12s

> These passages show current approaches for predicting gene perturbation responses include $D^{2}R^{2}$, DeMixPert, CPA, and GEARS [1], [2], [3], [4]. However, they do not provide a survey of current literature as the field is not covered by this fixed set of indexed papers.

---

### B026 — agrees

**Q:** Which methods currently provide the strongest evidence for perturbation prediction?

- label: `expect: scoped` (coverage: partial)
- topics matched: perturbation prediction=12
- system: **answer**, 2 citations, 14s

> Based on the findings presented in these passages, no single method performs well across all datasets. trVAE, CellOT, and inVAE demonstrate the best overall performance on single-condition datasets for cellular context generalization [1]. For perturbation generalization scenarios, GenePert is the optimal choice for predicting genetic sing…

---

### B027 — agrees

**Q:** How well do current gene perturbation models generalize across cell types and datasets?

- label: `expect: scoped` (coverage: partial)
- topics matched: perturbation prediction=12
- system: **answer**, 4 citations, 17s

> These passages show that current gene perturbation models have varying generalizability across different cell types and datasets [1], with no single method performing well across all datasets [1]. The generalizability is assessed in two key scenarios: cellular context generalization and perturbation generalization [2], using 14 and 18 met…

---

### B028 — agrees

**Q:** What evidence exists for experimentally validating computational gene perturbation predictions?

- label: `expect: scoped` (coverage: partial)
- topics matched: perturbation prediction=12
- system: **answer**, 1 citations, 9s

> [1] While these methods still require cell engineering, existing computational methods for predicting perturbation effects present their own limitations. Existing computational methods have not experimentally validated their predictions.

---

### B029 — agrees

**Q:** Which recent studies use held-out interventional biological data?

- label: `expect: scoped` (coverage: partial)
- topics matched: causal inference=9
- system: **answer**, 2 citations, 13s

> [1] Another recent study [2] showed that zero-shot ISP prediction performance (i.e., all perturbations are unseen) of Geneformer and scGPT lags behind GEARS, which is trained on perturbation data and tested on a held-out subset of that data.

---

### B031 — agrees

**Q:** What are the current research trends in causal inference for biology?

- label: `expect: scoped` (coverage: partial)
- topics matched: causal inference=9
- system: **refuse**, 0 citations, 11s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B033 — agrees

**Q:** What are the current research directions in computational pathology?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: pathology=0
- system: **refuse**, 0 citations, 12s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B034 — agrees

**Q:** What are the major open problems in AI for radiology?

- label: `expect: scoped` (coverage: partial)
- topics matched: medical imaging=8
- system: **answer**, 1 citations, 11s

> The major open problems in AI for radiology include rigorous external validation [1] and ensuring the models are generalizable and clinically reliable [1]. Additionally, addressing ethical issues such as data ownership, algorithmic bias, and patient consent [1] is crucial. Lastly, regulatory frameworks need to evolve to accommodate AI-dri…

---

### B035 — agrees

**Q:** What are the current trends in automated radiology report generation?

- label: `expect: scoped` (coverage: partial)
- topics matched: medical imaging=8, report generation=0
- system: **refuse**, 0 citations, 29s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B036 — agrees

**Q:** What are the major gaps in automated renal CT report generation?

- label: `expect: scoped` (coverage: thin)
- topics matched: renal imaging=1, report generation=0
- system: **answer**, 2 citations, 10s

> The major gaps in automated renal CT report generation are the scarcity of reports in public datasets, as only two models specifically target this task: M3D [1] (publicly released) and Merlin [2] (partially released).

---

### B037 — agrees

**Q:** What datasets are currently used for renal imaging and radiology-report generation?

- label: `expect: scoped` (coverage: partial)
- topics matched: medical imaging=8, renal imaging=1, report generation=0
- system: **refuse**, 0 citations, 10s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B038 — agrees

**Q:** What recent work addresses kidney stones, cysts, tumors, and renal measurements jointly?

- label: `expect: scoped` (coverage: thin)
- topics matched: renal imaging=1
- system: **answer**, 2 citations, 15s

> [1] A Clinically Interpretable Deep CNN Framework for Early Chronic Kidney Disease Prediction Using Grad-CAM-Based Explainable AI — LITERATURE REVIEW, p2-4 discusses kidney stones, cysts, and tumors. [2] A Clinically Interpretable Deep CNN Framework for Early Chronic Kidney Disease Prediction Using Grad-CAM-Based Explainable AI — LITERATU…

---

### B039 — agrees

**Q:** What are the strongest current benchmarks for medical report generation?

- label: `expect: scoped` (coverage: partial)
- topics matched: medical imaging=8, report generation=0, evaluation=3
- system: **refuse**, 0 citations, 14s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B040 — agrees

**Q:** What evidence shows that generated medical reports are clinically reliable?

- label: `expect: scoped` (coverage: partial)
- topics matched: medical imaging=8
- system: **answer**, 1 citations, 11s

> [1] Constructing 3D Image-Text Tumor Datasets RadGPT A V X — Experiment & Result, p6-7 demonstrates that generated medical reports are clinically reliable. An expert radiologist manually evaluated structured reports generated by the models. The radiologist deemed 75.6% of the tumors reported by true-positives, and 93.5% of them were corre…

---

### B041 — agrees

**Q:** Identify recent papers that challenge a widely accepted claim in a research field.

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **refuse**, 0 citations, 10s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B042 — agrees

**Q:** Identify recent papers that replicate or fail to replicate an influential result.

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **refuse**, 0 citations, 11s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B043 — agrees

**Q:** What research trends have emerged in the last 2–3 years in a selected field?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **refuse**, 0 citations, 10s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B044 — agrees

**Q:** Which research directions appear to be declining, and what evidence supports that conclusion?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **refuse**, 0 citations, 9s

> There is no evidence in the passages that support any research directions are declining. The passages discuss ongoing challenges and future research directions in various fields including language models, federated learning, diabetic retinopathy detection, gene regulatory network inference, and privacy-preserving federated learning.

---

### B048 — agrees

**Q:** Which gaps appear important but remain weakly addressed?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **refuse**, 0 citations, 10s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B051 — agrees

**Q:** What methods currently outperform simple linear or classical baselines in gene perturbation prediction?

- label: `expect: scoped` (coverage: partial)
- topics matched: perturbation prediction=12
- system: **answer**, 2 citations, 12s

> None of the methods currently outperform simple linear or classical baselines in gene perturbation prediction [1], [2].

---

### B055 — agrees

**Q:** What diseases are currently receiving the most attention in AI-driven biomedical research?

- label: `expect: scoped` (coverage: partial)
- topics matched: medical imaging=8
- system: **refuse**, 0 citations, 12s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B057 — agrees

**Q:** Which biological datasets are becoming standard benchmarks?

- label: `expect: scoped` (coverage: thin)
- topics matched: evaluation=3
- system: **answer**, 1 citations, 11s

> [1] A large-scale benchmark for network inference from single-cell perturbation data includes two different biological contexts (cell lines RPE-1 and K562) as a publicly available dataset.

---

### B058 — agrees

**Q:** What are the major reproducibility problems in current computational biology research?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **refuse**, 0 citations, 13s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B060 — agrees

**Q:** What recent evidence exists for dataset leakage or benchmark contamination?

- label: `expect: scoped` (coverage: thin)
- topics matched: evaluation=3
- system: **answer**, 5 citations, 12s

> These passages show recent evidence for dataset leakage and benchmark contamination. [1] discusses contamination as a growing problem, [2] and [3] detail methods to measure and prevent contamination, and [4] and [5] provide examples of contamination found in benchmarks.

---

### B061 — agrees

**Q:** What are the strongest current methods for evaluating scientific AI systems?

- label: `expect: scoped` (coverage: thin)
- topics matched: evaluation=3
- system: **refuse**, 0 citations, 13s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B062 — agrees

**Q:** How are researchers currently evaluating AI-generated scientific claims?

- label: `expect: scoped` (coverage: thin)
- topics matched: evaluation=3
- system: **refuse**, 0 citations, 8s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B063 — agrees

**Q:** What methods are used to detect unsupported or hallucinated scientific claims?

- label: `expect: scoped` (coverage: thin)
- topics matched: hallucination=1
- system: **answer**, 1 citations, 8s

> [1] Effective methods involve training reward models to discriminate between desirable and undesirable outputs.

---

### B064 — agrees

**Q:** Which current benchmarks test whether AI systems distinguish evidence from speculation?

- label: `expect: scoped` (coverage: thin)
- topics matched: evaluation=3
- system: **refuse**, 0 citations, 12s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B065 — agrees

**Q:** How well do current research assistants perform on multi-paper synthesis?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **refuse**, 0 citations, 10s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B066 — agrees

**Q:** What are the current limitations of AI systems for literature review?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **refuse**, 0 citations, 10s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B068 — agrees

**Q:** Can current AI systems distinguish prediction from measurement in scientific papers?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **refuse**, 0 citations, 10s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B069 — agrees

**Q:** Can current AI systems distinguish observational from interventional evidence?

- label: `expect: scoped` (coverage: partial)
- topics matched: causal inference=9
- system: **refuse**, 0 citations, 9s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B070 — agrees

**Q:** What are the current best practices for evidence-grounded scientific question answering?

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **refuse**, 0 citations, 9s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B073 — agrees

**Q:** Compare the conclusions of recent papers that disagree on an important research question.

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **refuse**, 0 citations, 10s

> I could not find sufficient evidence in the indexed papers to answer that.

---

### B080 — agrees

**Q:** Given a research field, synthesize its current state, major trends, unresolved gaps, strongest evidence, and promising future opportunities using recent external literature.

- label: `expect: refuse` (coverage: beyond_corpus)
- topics matched: none
- system: **refuse**, 0 citations, 14s

> I could not find sufficient evidence in the indexed papers to answer that.
