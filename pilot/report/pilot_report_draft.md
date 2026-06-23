# Adapting a Six-Metric XAI Evaluation Framework to a Construction-Safety VLM — Pilot Report

**Draft, Day 12 of the two-week pilot.** Every number below traces to a specific CSV or figure produced in Days 1-11 (`pilot/results/`); none are invented or estimated for this writeup. Citations in parentheses point to the source file.

## 1. Motivation

Professor Mustafa Abdallah's six-metric XAI evaluation framework (Descriptive Accuracy, Sparsity, Stability, Efficiency, Robustness, Completeness) was originally built and validated for tabular network-intrusion DNNs. This pilot asks one question ahead of a collaboration meeting: **does the same six-metric framework transfer to a Vision-Language Model making visual safety judgments**, or does crossing from tabular features to image+text inputs break the framework's assumptions in ways that need to be solved before a full study is worth committing to?

We chose a construction-site safety-violation VQA task (hard hat / harness / guardrail / excavator-proximity) as the test bed: it has clear "ground truth" safety rules, a public annotated dataset, and a small open VLM (Florence-2) that runs on a single consumer GPU — a realistic, low-cost feasibility test rather than a polished benchmark result.

## 2. Dataset

[`LouisChen15/ConstructionSite`](https://huggingface.co/datasets/LouisChen15/ConstructionSite) ("ConstructionSite 10k"), Hugging Face, `cc-by-nc-4.0`, gated (access already confirmed for this account). 10,013 images total — 7,009 train / 3,004 test. We worked exclusively from the **test split**.

Schema (verified directly against the dataset, not assumed from the proposal docs): `image, image_id, image_caption, illumination, camera_distance, view, quality_of_info, rule_1_violation..rule_4_violation` (each a bounding box + reason, or `null` if compliant), plus `excavator`, `rebar`, `worker_with_white_hard_hat` grounding boxes. Rule-to-hazard-class mapping, read off the dataset card:

| Rule | Hazard concept | Pilot class |
|---|---|---|
| rule_1 | Missing basic PPE (hard hat / vest / eye protection) | `ppe_violation` |
| rule_2 | No harness at height ≥3m without edge protection | `fall_hazard` |
| rule_3 | No guardrail/edge protection at height or excavation ≥3m | `fall_hazard` |
| rule_4 | Worker in excavator blind spot/operating radius | `struck_by_risk` |
| (all null) | — | `compliant` |

**Sample**: a balanced draw of 50/class (200 target) was attempted (`scripts/02_select_samples.py`, seed 42). The test split could only supply **13** `struck_by_risk` examples, not 50 — confirmed directly against `data/pilot_samples.csv`'s actual value counts (`ppe_violation 50, fall_hazard 50, compliant 50, struck_by_risk 13`), not assumed. **The pilot's actual n is 163, not 200**, and every `struck_by_risk` number in this report rests on n=13.

## 3. Model and the Grounding-Proxy Adaptation

`microsoft/Florence-2-base-ft`, 231.6M params, MIT license. Two real technical gaps surfaced on Day 1 that the original proposal docs hadn't accounted for, and both required design decisions rather than just following the plan literally:

**Florence-2 has no native free-form VQA task token.** Its task vocabulary is grounding/captioning tokens (`<CAPTION>`, `<OD>`, `<OPEN_VOCABULARY_DETECTION>`, `<CAPTION_TO_PHRASE_GROUNDING>`, etc.) — there is no `<VQA>` token to literally ask "Is the worker wearing a hard hat?" as the source proposal envisioned. **Adaptation**: each safety rule was reframed as a grounding query, not a yes/no question. Rules 1-3 (PPE/guardrail) call `<OPEN_VOCABULARY_DETECTION>` for "worker" and the rule's safety-object phrase independently, then answer **violation** if any detected worker has no nearby safety-object box (`regions.all_boxes_covered`, proximity threshold 8% of image dimension for body-worn PPE, 20% for the structurally-protective guardrail). Rule 4 detects `worker` and `excavator` independently and checks spatial proximity directly — a relational check, not an existence check. This is stated here explicitly as an adaptation of the VQA framing, not hidden inside the implementation.

**The proxy's first version was nearly useless, and was redesigned mid-pilot (Day 3).** The first, literal "does a hard-hat box exist anywhere in the scene" version answered "compliant" for almost every image — **3.0% sensitivity** (3/100 violations caught), 97.4% specificity (`results/day3_findings.md`) — because most scenes contain multiple workers, and one compliant worker is enough to satisfy a scene-level existence query even when a different worker in the same frame is the one the dataset flagged. This was diagnosed, not patched over, and the proxy was redesigned to a **person-relative** check (described above): each detected worker individually needs a nearby safety object. Sensitivity improved **9x to 27.0%** (specificity fell to 73.7%, an expected precision/recall tradeoff for a proxy that now actually discriminates instead of defaulting to "compliant"). Per-rule sensitivity after the redesign: rule_1 16% (8/50), rule_2 30.8% (4/13), rule_3 40.5% (15/37). **A residual, documented limitation**: Florence-2's `<OPEN_VOCABULARY_DETECTION>` for the singular phrase "worker" returns only one box even in multi-worker scenes — visually confirmed on two images with 5+ visible workers — capping how much further the person-relative redesign can improve sensitivity without a different detection strategy (e.g. `<DENSE_REGION_CAPTION>` for exhaustive instance enumeration), not pursued in this pass.

**Confidence proxy**: mean per-token generation probability from `model.generate(..., output_scores=True)` — the same hedged "confidence proxy" language the source proposal already used, not a calibrated probability.

## 4. Methods Summary

All six metrics ran over the same 163-sample pilot set, using Day 4's `standardize_regions()` to produce a ranked list of "explanation candidate regions" per sample (the model's own returned grounding box(es), ranked by area descending; a 4×4 grid fallback only when zero boxes were returned at all — 4/163 samples, 2.5%).

| Metric | What it tests | Method |
|---|---|---|
| Descriptive Accuracy | Does masking the top-ranked region change the answer? | Black-fill mask top-1 (then top-1+top-2), rerun, compare answers (Day 5) |
| Visual Sparsity | Is the visual explanation concentrated or scattered? | Decoder→encoder cross-attention heatmap over the 24×24 image-patch grid (Day 6) |
| Stability | Is the explanation reproducible? | 3 sampled reruns (`do_sample=True, temperature=0.7`) vs. one deterministic beam-search reference (Day 7) |
| Efficiency | What's the GPU-time budget at scale? | Aggregate per-sample timings already logged in Days 3/5/6/7, linearly extrapolated to n=1000 (Day 8) |
| Robustness | Does the explanation survive noise? | 4 image perturbations (blur, low_light, occlude, contrast_shift) + reworded-prompt variants + a bounded 20-image synthetic-patch stretch test (Day 9) |
| Bounded Completeness | Is the top-ranked region actually necessary? | Bucket each sample using Day 5's masking result: `no_usable_explanation` (grid fallback) / `explanation_weak` (masking didn't change the answer) / `explanation_supported` (it did) (Day 10) |

Day 6 deviated from the plan's literal "attention rollout" instruction after checking Florence-2's actual modeling code: classic ViT-style rollout assumes a single-modality self-attention stack, but Florence-2's encoder runs one shared stack over a fused image+text sequence, making rollout's recursive mixing ill-defined here. Decoder→encoder cross-attention was used instead — well-defined per generation step, no rollout approximation needed.

## 5. Results

(`results/pilot_metric_summary.csv`, `results/figures/summary/metric_summary_by_class.png` — Day 11 rollup of Days 5-10's per-sample CSVs.)

| Metric | Overall | compliant (n=50) | ppe_violation (n=50) | fall_hazard (n=50) | struck_by_risk (n=13) |
|---|---|---|---|---|---|
| Descriptive accuracy (top-1 mask flips answer) | 36.2% | 28.0% | 28.0% | 46.0% | 61.5% |
| Visual sparsity (top-5-cell attention mass, of 576 cells) | 0.108 | 0.099 | 0.127 | 0.109 | 0.067 |
| Stability (3-run answer agreement) | 77.5% | 74.7% | 76.0% | 78.7% | 89.7% |
| Robustness, Level 1 (answer survives image perturbation) | 80.7% | 79.5% | 86.0% | 75.0% | 86.5% |
| Bounded completeness (`explanation_supported` rate) | 43.6% | 32.0% | 36.0% | 58.0% | 61.5% |
| Efficiency, full pipeline @ n=1000 | ~0.5 GPU-hours (~30 min), single RTX 3070 | — | — | — | — |

These headline numbers are necessary but not sufficient — the real value of this pilot is in *why* they land where they do, below.

## 6. Cross-Cutting Findings

### 6.1 The framework surfaced a genuine adaptation bug in the proxy itself (Day 5)

Masking *more* evidence should never un-flip an answer that masking less already flipped — yet top-1+top-2 masking (35.0%) scored slightly *lower* than top-1 alone (36.2%), regressing in 15/163 samples. Traced to a real interaction with the proxy's own fallback logic: masking the top-ranked region (almost always the *worker*, see 6.2) doesn't just remove evidence, it removes the precondition (`if not worker_boxes`) the person-relative check needs to run at all, silently rerouting to the cruder scene-level fallback (`"compliant" if object_boxes else "violation"`). That fallback's answer then depends only on whether the *other* region is still visible — independent of how much area was masked. This same fallback-rerouting mechanism reappears under image perturbations (Day 9: 20% of occlude's answer-flips co-occur with a lost worker detection) and inflates a small slice of completeness verdicts (Day 10: 3 of 71 `explanation_supported` cases, dropping the genuinely-supported count to 68/163 = 41.7%). This is exactly the kind of "the explanation method works correctly but the upstream task design has a sharp edge" finding the framework is supposed to surface — it did.

### 6.2 An area-based region-ranking heuristic systematically disadvantages the smallest safety-relevant objects

Day 4's region ranking (no detection-class signal available, by design — model boxes are ranked by raw area) puts the *worker's* whole-body box ahead of the much smaller PPE item in 96% of `ppe_violation`/rule_1 samples, vs. only 24% for rule_3 (guardrail, often itself the largest object in frame). This single mechanism explains three separate metrics' weakest results all landing on PPE: completeness is 33.3% supported for rule_1 vs. 55.1% for rule_3 (Day 10); stability's safety-object-box IoU is 0.499 for rule_1's hard hat vs. 0.922 for rule_4's excavator (Day 7); and there's a related but independent confound in sparsity, where a small object's 576-cell attention grid mechanically concentrates regardless of reasoning quality (correlation of -0.70 between log(box area) and top-5-cell attention mass, Day 6). **The irony, worth saying plainly to Professor Abdallah**: hard hats and harnesses are exactly the PPE items where a stable, complete explanation matters most for real-world trust, and they are the least stable and least complete by every one of these measures — for a mechanical, fixable reason (the ranking heuristic), not because the model's underlying reasoning about PPE is worse than its reasoning about excavators.

### 6.3 A stable answer can hide a fully-drifted explanation

Quantified directly (Day 9): of 611 robustness reruns where the final answer didn't change, **28.3% still show the model's safety-object box moving to something essentially unrelated** (IoU < 0.3) — most severely under blur (58.8%) and reworded prompts (56.7%). An answer-level robustness number alone would call these samples perfectly robust; they aren't, at the explanation level. This is the strongest single argument in this pilot for reporting answer-level and explanation-level robustness as two separate numbers, never blended into one score — and it generalizes the same lesson Day 6 already established for sparsity (visual vs. text attribution must stay separate, not merged).

### 6.4 Sparsity and completeness measure genuinely different, sometimes opposite things

`struck_by_risk` has the *highest* descriptive accuracy and completeness of any class (61.5%/61.5%) but the *lowest* visual sparsity (0.067 vs. 0.099-0.127 elsewhere) — its explanation is the least visually concentrated, yet the most load-bearing by the masking test (Day 11, consistent with Day 6's box-area confound: excavators are large, so cross-attention spreads over more of the 24×24 grid even though masking the whole region still reliably breaks the proximity check). Concentration and load-bearing-ness are not proxies for each other.

### 6.5 Efficiency is not a blocker

Summing the full six-step pipeline's logged per-sample timings and extrapolating linearly to n=1000 gives **~0.5 GPU-hours (~30 minutes)** on a single RTX 3070 (Day 8). There is no compute obstacle between this pilot and a full 3,004-image or larger study.

## 7. Limitations (explicit, not papered over)

- **Sample size and composition**: 163 samples, not the planned 200 — the test split only contains 13 `struck_by_risk` examples, not 50. Every `struck_by_risk` number in this report is far noisier than the other three classes' n=50 and should be read as directional, not a stable estimate. No statistical significance testing was performed at this n; that's an explicit ask for Professor Abdallah's guidance (Section 8).
- **Robustness is Level 1 (image perturbations) plus a partial Level 2 (prompt rewording)**, not the full Level 3 envisioned in the source proposal. `rule_4` has no second prompt phrasing and was excluded from Level 2 rather than silently scored as 0% change. The Level 3 synthetic-patch test was explicitly scoped as a bounded, time-boxed 20-image stretch task, not a full adversarial study — and even within that scope, the real usable signal was n=4 (most candidates already had a baseline proxy answer that made "fooling" undefined), with a real but weak result (3/4 flipped) that doesn't show a clean placement-to-flip correlation.
- **Completeness is bounded** (top-1/top-2 region masking only, per `technical-limitations-take-and-mitigation.md`'s explicit scope), not full completeness over the entire region/feature space, and not run on the full 3,004-image test split.
- **No LRP or DeepLIFT.** Both assume architectures these tools were never built for; cross-attention extraction was substituted for attention rollout after directly verifying rollout's assumptions don't hold for Florence-2's fused image+text encoder (Section 4). Whether this substitution is an academically acceptable stand-in for LRP/DeepLIFT-style attribution is exactly the kind of question this pilot wants Professor Abdallah's judgment on.
- **The grounding-based VQA proxy is an adaptation, stated as such**, not a literal implementation of the "ask a free-form safety question" framing in the original proposal — Florence-2 has no task token that supports that framing at all.
- **Confidence is an uncalibrated proxy** (mean token-generation probability). Day 9 found it can move in counter-intuitive directions (confidence *rising* under occlusion, alongside a high answer-change rate), and Day 10 separately confirmed it carries no discriminative signal for completeness classification at this sample size — it is reported, but was not used as a load-bearing signal in the bounded-completeness classification for that reason.
- **Day 8's per-sample timing numbers for cross-attention extraction and stability sampling rest on a 15-sample calibration**, not the full 163-sample run (unlike Days 3/5's exact aggregation) — good enough for an order-of-magnitude efficiency story, should be re-measured with direct instrumentation before quoting precise numbers in anything more formal.

## 8. Next Steps / Where We Want Professor Abdallah's Guidance

1. **LRP/DeepLIFT feasibility on transformer/VLM architectures generally** — is decoder→encoder cross-attention an acceptable substitute in the framework's terms, or does a transformer-native attribution method need to be added to the framework's toolkit rather than treated as a one-off pilot workaround?
2. **Statistical rigor at small n** — confidence intervals / significance testing across classes and rules, especially for the 13-sample `struck_by_risk` class, before any number here is presented as a stable estimate.
3. **A better region-ranking heuristic** for completeness/stability — rank by "is this the rule's queried object class" before falling back to raw area, to remove the worker-vs-object confound documented in Section 6.2.
4. **Scaling to the full 3,004-image test split** — Section 6.5 shows compute is not the obstacle; what we'd want is guidance on whether 3,004 is the right target or whether a stratified larger sample (to fix the `struck_by_risk` shortage) is more valuable than raw scale.
5. **A real Level 3 robustness study** — the current 20-image, n=4-meaningful synthetic patch test is a plausibility check, not a result; a properly powered adversarial-patch study (optimized placement, larger n) is a natural full-study extension.
6. **Revisit rule_2's two prompt phrasings** ("safety harness" vs. "fall-protection lanyard") — Day 9 found these aren't synonyms (a harness is a worn garment, a lanyard is its tether) and may be grounding two different physical objects, not two wordings of one concept; worth resolving before rule_2's robustness numbers are trusted in a larger study.
