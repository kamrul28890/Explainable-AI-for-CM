# 5-Minute Talking Points — Meeting with Professor Abdallah

Speaking notes, not a script. Full detail is in `pilot_report_draft.md`; numbers here are pulled from there. Visuals to have open: `key_visualizations.md`.

## The question (30s)

Your six-metric framework (Descriptive Accuracy, Sparsity, Stability, Efficiency, Robustness, Completeness) was built and validated on tabular intrusion-detection DNNs. Does it transfer to a Vision-Language Model making visual judgments — or does crossing from tabular features to image+text inputs break it? That's the one question this two-week pilot was built to answer.

## What was tested (1 min)

- Construction-site safety VQA: hard hat, harness, guardrail, excavator-proximity — four rules mapped to four hazard classes.
- `LouisChen15/ConstructionSite` on Hugging Face, 163 images sampled from its 3,004-image test split (50/class, except `struck_by_risk` which only has 13 in the whole split).
- Florence-2-base-ft, 231M params, single RTX 3070 — deliberately small and cheap, a feasibility test, not a benchmark run.
- All six metrics ran end-to-end over all 163 samples. Pipeline reruns cleanly from a brand-new venv on a fresh subset with zero manual intervention (verified Day 13) — this isn't a one-off that happened to work on one machine.

## What worked (1.5 min)

- The framework's six metrics all had a meaningful adaptation path to a VLM, none had to be abandoned.
- The biggest real obstacle wasn't the framework — it was that **Florence-2 has no native VQA token**. Reframed each rule as a grounding query instead of literal Q&A. The first version of that reframing barely worked (3.0% sensitivity — answered "compliant" almost regardless of ground truth); redesigned mid-pilot to a person-relative check, **9x sensitivity gain (27.0%)**. Worth saying plainly: this is the kind of model-level gap the framework itself doesn't catch — Descriptive Accuracy and Completeness measure the *explanation*, not whether the underlying answer-generation method is sound. We had to catch that ourselves.
- Compute is a non-issue: ~30 minutes of GPU time covers the full six-metric pipeline at 1,000 samples.
- LRP/DeepLIFT don't apply to this architecture as-is; decoder→encoder cross-attention substituted cleanly as the attribution source after verifying (not assuming) that classic attention rollout's single-modality assumption doesn't hold for Florence-2's fused image+text encoder.

## What failed / what's more interesting than the headline numbers (1.5 min)

- **One mechanical choice — ranking explanation regions by raw area — explains three separate metrics' weakest results all landing on PPE** (hard hat, harness): the worker's whole-body box is almost always larger than the PPE item, so "the top region" usually means the worker, not the hard hat. Fixable, but real, and it means the metrics make hard hats and harnesses look like the model's worst-explained categories for a reason that has nothing to do with the model's actual reasoning about PPE.
- **A stable answer can hide a fully-drifted explanation.** 28.3% of robustness reruns where the final answer didn't change still show the evidence box moving to something unrelated. If we'd only reported answer-level robustness, we would have missed this completely.
- **A worker-detection fallback path recurs across three different days' tests** (masking, perturbation, completeness) and quietly inflates a slice of every metric that touches it — traced to one specific code branch each time, not left as unexplained noise.
- **The synthetic hard-hat patch test (a small stretch task) gave a genuinely counterintuitive result**: a well-placed patch failed to fool the model, a badly-placed one succeeded. n=4 usable cases — not a result, a "worth a bigger study" signal.

## Where we want your guidance (30s)

1. Is decoder→encoder cross-attention an acceptable LRP/DeepLIFT substitute for transformer VLMs, or does the framework need a transformer-native attribution method?
2. Statistical rigor at small n, especially the 13-sample `struck_by_risk` class.
3. Full-dataset scaling (3,004 images) vs. fixing the class imbalance first — which is the better next move?
4. A properly powered Level-3 adversarial-patch study — our 4-sample stretch test just shows it's worth doing properly.
