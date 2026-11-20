**Subject:** Pilot complete — six-metric XAI framework on a construction-safety VLM (findings, proposed scale-up, and a request to meet)

Dear Professor Abdallah,

I'm writing to update you on the technical pilot and to share a full package ahead of a meeting I'd like to request.

**What we set out to test.** Whether your six-metric XAI evaluation framework — Descriptive Accuracy, Sparsity, Stability, Efficiency, Robustness, and Completeness, originally built and validated on tabular network-intrusion DNNs — transfers to a Vision-Language Model making visual safety judgments. We ran a two-week feasibility pilot with Florence-2-base-ft (231M params, a single RTX 3070) on 163 images drawn from the ConstructionSite dataset.

**Headline result: yes, the framework transfers — but it required two real, quantified adaptations, not zero changes.**
1. Florence-2 has no native visual-question-answering token, so each safety rule had to be reframed as an open-vocabulary grounding query rather than a literal yes/no question.
2. Our first version of that proxy barely discriminated (3.0% sensitivity — effectively a constant "compliant" classifier). A person-relative redesign, mirroring the dataset's own per-worker labels, produced a **9× sensitivity gain (to 27.0%)**. All six metrics then ran end-to-end and produced genuine, interpretable signal.

**What we found that matters more than the individual numbers.** The pilot surfaced a small number of cross-cutting mechanisms rather than a list of isolated results — most importantly:
- One upstream design choice (ranking candidate explanation regions by box area) is the single highest-leverage issue: it systematically tests the worker's body box instead of the small PPE item, weakening three metrics at once specifically on the PPE rules we care about most.
- A stable final answer can sit on top of an explanation that has completely drifted (28.3% of stable-answer cases) — direct evidence for your thesis that answer-level metrics are necessary but not sufficient.
- Most consequentially for the next phase: on Florence-2 the safety judgment is ultimately computed by our own geometric code, not by the model itself. This means some metrics partly measure our proxy rather than the model's reasoning. I want to be candid about this because it directly shapes what I'm proposing next.

**What I'm proposing for the scale-up.** Rather than simply scaling the sample count from 163 to the full 3,004-image split, I recommend we first fix the foundations (region-ranking, class labeling, decoding consistency), then add a **native-VQA model (Qwen2.5-VL)** so the framework can be evaluated against a model that makes the judgment itself — removing the proxy confound — and finally extend to the SODA and CMA datasets. The attached implementation plan lays this out phase by phase.

**The package (attached):**
- **Project report (PDF)** — the complete pilot, aggregated: motivation, method, all six metrics with results and figures, cross-cutting findings, limitations, and reproducibility.
- **Presentation (PPTX)** — a detailed, self-contained walk-through of every step with visualizations, for the meeting.
- **Scale-up implementation plan** — a detailed, phase-by-phase engineering plan with all architecture, pipeline, and hyperparameter choices.

**Six decisions I'd value your judgment on** (detailed in the package): (1) whether decoder→encoder cross-attention is an acceptable stand-in for LRP/DeepLIFT or a transformer-native method is required; (2) minimum-n and statistical testing standards, especially for the scarce struck_by_risk class; (3) approving the region-ranking fix; (4) full-dataset scaling vs. fixing class imbalance first; (5) scope for a properly powered adversarial-robustness study; (6) treating the two rule_2 phrasings as distinct concepts.

Could we schedule a short meeting in the coming week to walk through the deck and lock these decisions before I begin the scale-up? I'm happy to work around your availability.

Thank you for your guidance.

Best regards,
[Your Name]
