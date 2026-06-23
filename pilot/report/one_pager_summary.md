# One-Pager: Construction-VLM XAI Pilot

**For:** meeting with Professor Mustafa Abdallah · **Status:** two-week feasibility pilot complete

## The question

Does Professor Abdallah's six-metric XAI evaluation framework (Descriptive Accuracy, Sparsity, Stability, Efficiency, Robustness, Completeness — built for tabular intrusion-detection DNNs) transfer to a Vision-Language Model making visual judgments? We tested this on a construction-safety VQA task with Florence-2-base-ft (231M params, single RTX 3070) over 163 images from [`LouisChen15/ConstructionSite`](https://huggingface.co/datasets/LouisChen15/ConstructionSite).

## Headline answer: yes, with two real adaptations, not zero changes

1. **Florence-2 has no native VQA token.** We reframed each safety rule as a grounding query (`<OPEN_VOCABULARY_DETECTION>` for "worker" + the safety object, person-relative coverage check) instead of literal Q&A.
2. **The first version of that proxy barely worked** (3.0% sensitivity — answered "compliant" almost regardless of ground truth). Redesigned mid-pilot to a person-relative check: **9x sensitivity gain (27.0%)**.

## The six metrics ran end-to-end. Headline numbers (163 samples):

| Metric | Result |
|---|---|
| Descriptive accuracy (masking top region flips the answer) | 36.2% |
| Visual sparsity (top-5-of-576-cell attention mass) | 0.108 |
| Stability (3-run sampled answer agreement) | 77.5% |
| Robustness, Level 1 (survives blur/low-light/occlusion/contrast) | 80.7% |
| Bounded completeness (top region masking is load-bearing) | 43.6% |
| Efficiency, full pipeline @ n=1000 | ~0.5 GPU-hours |

## What we found that matters more than the headline numbers

- **One mechanical region-ranking choice (rank by area) explains three metrics' weakest results all landing on PPE** (hard hat/harness): the worker's body box is almost always larger than the PPE item, so masking/tracking "the top region" usually means the worker, not the hard hat. Fixable with a smarter ranking rule.
- **A stable answer can hide a fully-drifted explanation**: 28.3% of robustness reruns where the answer didn't change still show the model's evidence box moving to something unrelated. Reporting answer-level robustness alone would have missed this entirely.
- **A worker-detection fallback path recurs across three different days' tests** (masking, perturbation, completeness) and inflates a small slice of every metric that relies on it — found, traced to a specific code branch, and quantified each time rather than left as unexplained noise.
- **Compute is not a blocker.** ~30 minutes of GPU time covers the entire six-metric pipeline at 1,000 samples.

## Where we want your guidance

1. Is decoder→encoder cross-attention an acceptable LRP/DeepLIFT substitute for transformer VLMs, or does the framework need a transformer-native attribution method?
2. Statistical rigor at small n — especially `struck_by_risk`, which only has 13 examples in the dataset's entire 3,004-image test split.
3. Is scaling to the full 3,004-image split the right next step, or is fixing the `struck_by_risk` class-imbalance more valuable than raw scale?
4. A properly powered Level-3 adversarial-patch study (our 20-image stretch test had only 4 meaningfully-testable candidates).

## Honest limitations (full list in `pilot_report_draft.md`)

163/3,004 samples; Level 1 + partial Level 2 robustness only; bounded (not full) completeness; no LRP/DeepLIFT; uncalibrated confidence proxy; the grounding-based VQA proxy is a stated adaptation, not literal free-form Q&A. **Reproducibility verified**: full pipeline reruns cleanly from a brand-new venv on a fresh 20-sample subset with zero manual intervention (Day 13).
