# Scale-Up Implementation Plan — Construction-VLM XAI Evaluation

**Version:** 1.0 · **Status:** draft for review · **Prerequisite:** two-week Florence-2 pilot complete (163 samples, all six metrics ran end-to-end).

This document is the engineering specification for scaling the pilot into a full study. It is written to be implementable by an engineer or agent **without prior context on the pilot**. It contains no code, but it specifies every architecture, pipeline, data, and hyperparameter choice needed to build the system. Where a value is a deliberate decision rather than an arbitrary default, the rationale is given. Items marked **[DECISION]** require Professor Abdallah's sign-off before implementation; a recommended default is provided for each so work is not blocked.

---

## 0. Orientation: what exists today, and the core problem to fix

**The pilot pipeline** is an 11-script sequence: environment check → balanced sample selection → baseline inference → region standardization → the five per-sample metrics (descriptive accuracy, sparsity, stability, robustness) plus efficiency → bounded completeness → cross-metric rollup. It runs on Florence-2-base-ft against the `LouisChen15/ConstructionSite` Hugging Face dataset.

**The four safety rules** the pipeline evaluates:
- `rule_1` — basic PPE present (hard hat / hi-vis vest) → class `ppe_violation`
- `rule_2` — fall protection (harness/lanyard) at height → class `fall_hazard`
- `rule_3` — edge protection (guardrail) at height/excavation edges → class `fall_hazard`
- `rule_4` — worker inside excavator blind-spot/swing radius → class `struck_by_risk`
- (no violation on any rule → class `compliant`)

**The single most important fact for this plan:** on Florence-2, the safety answer is **not produced by the model**. Florence-2 only returns bounding boxes for phrases ("worker", "hard hat", ...). The compliant/violation judgment is then computed by hand-written geometric Python (per-worker coverage within a fixed pixel-distance threshold). Consequently, several XAI metrics partly measure the sensitivity of that geometric proxy rather than the model's own reasoning. **The scale-up's central scientific goal is to make the metrics measure model faithfulness**, which requires (a) fixing the proxy's known biases and (b) adding a model that makes the judgment itself (native-VQA). Everything below serves that goal.

---

## 1. Guiding principles (apply to every phase)

1. **Freeze the pilot record.** All new behavior lands behind configuration flags. The recorded 163-sample pilot numbers must remain reproducible byte-for-byte. Never mutate a metric's default behavior in place; add a new mode and make the old one the default until the new one is validated.
2. **Every number traces to a results file.** No figure or table value is ever hand-typed into a report; report artifacts are generated programmatically from the results CSVs at build time.
3. **Parametrize everything that could change with scale.** No sample count, class size, image resolution, grid dimension, or model ID may be hardcoded in analysis or plotting code. (The pilot shipped a bug where `n=163` and class sizes were literal strings; that class of bug must be designed out.)
4. **Separate model-faithfulness signal from proxy-logic signal.** Whenever a metric's result could be produced by the geometric proxy's own branching rather than the model, instrument for it and report the corrected number alongside the raw one.
5. **Report the metric's true target, never only the ranking heuristic's pick.** For every region-based metric, report both the ranked top region and the rule's actual queried object.
6. **Determinism where it matters, controlled stochasticity where it's the point.** Unify decoding across metrics (see Phase 2). Use multiple seeds for anything sampling-based.

---

## 2. Phase 0 — Decisions to lock before build (with recommended defaults)

These four decisions change how much of Phases 2–5 gets built. Recommended defaults let Phase 1 proceed in parallel.

| # | Decision | Recommended default | Consequence if changed |
|---|---|---|---|
| D1 | **Attribution method.** Accept decoder→encoder cross-attention, or add a transformer-native gradient method (Integrated Gradients / DeepLIFT / a transformer LRP variant)? | **Add Integrated Gradients (IG) via Captum as a second, independent attribution stream**, keeping cross-attention as the primary. Report agreement between the two. | If gradient methods are required, model must run in **bf16 or fp32** for attribution (fp16 underflows gradients), and attribution compute grows 10–50×. |
| D2 | **Model scope.** Florence-2 only, or add a native-VQA model? | **Add Qwen2.5-VL (3B and 7B)** as native-VQA comparison; keep Florence-2 as the region-native baseline; one frontier model (e.g. GPT-4o) as benchmark-only reference. | Determines whether metrics measure model faithfulness (native-VQA) or proxy sensitivity (Florence-2 only). |
| D3 | **Class labeling.** Keep mutually-exclusive priority collapse, or move to multi-label? | **Multi-label.** An image can be `ppe_violation` AND `struck_by_risk` simultaneously. | Recovers the scarce `struck_by_risk` class (13 → ~70 across the full dataset) and enables multi-hazard completeness testing the proposal promised. |
| D4 | **Detection task token.** Keep single-box `<OPEN_VOCABULARY_DETECTION>`, or move to exhaustive instance enumeration? | **Evaluate `<DENSE_REGION_CAPTION>` / `<OD>` on a validation subset first; adopt only if it measurably improves multi-worker recall.** Do not assume any token "guarantees" full enumeration — measure it. | Multi-worker detection is a precondition for valid completeness and for rule_4 (proximity) correctness; but longer outputs raise decode cost and make beam-vs-greedy matter (see Phase 5 efficiency). |

**Minimum-n / statistics standard [DECISION, deferred to D2/D3 outcome]:** set a minimum samples-per-class before a metric is reported as a stable estimate (recommend n ≥ 30 per class per rule as a floor), and adopt a nonparametric significance test (Wilcoxon signed-rank, matching Prof. Abdallah's prior IDS work) plus bootstrap confidence intervals on every headline number.

---

## 3. Phase 1 — Foundational correctness fixes (highest leverage, lowest risk)

These are cheap, require no new models, and unblock six downstream chapters. Do these first. All land behind flags; default off until validated, then flipped on for the scale run.

### 3.1 Rule-aware region ranking *(fixes the pilot's #1 issue)*
- **Problem:** candidate explanation regions are ranked by pixel area, so the worker's body box outranks the small PPE object in ~96% of PPE samples; every masking/overlap metric then tests the wrong box.
- **New ranking policy:** rank candidate regions by a **semantic-match-first** key: (1) regions whose label matches the rule's queried object class rank above all others; (2) within a tier, rank by a real importance signal if available (cross-attention mass inside the box — see Phase 4), else by area as a last resort.
- **Object-class map:** rule_1 → {hard hat, hi-vis vest}; rule_2 → {safety harness, lanyard}; rule_3 → {guardrail, edge barrier}; rule_4 → {excavator} (proximity partner) with worker as the subject.
- **Config:** `region_ranking ∈ {area, rule_aware, attention}`; default `area` (frozen pilot), scale run uses `rule_aware`.
- **Validation gate:** re-run descriptive accuracy and completeness; confirm the previously-measured 11.9-percentage-point "mask worker vs mask object" gap narrows substantially. Record before/after.

### 3.2 Multi-label classification + class-balanced sampling *(fixes struck_by_risk scarcity)*
- **Problem:** priority-collapse assigns each image exactly one class, discarding the 11 of 24 real `struck_by_risk` test-split images that also had a higher-priority violation. This is a labeling artifact, not a dataset property.
- **Fix:** classification returns the **full set** of violated classes per image (it is already computed internally, then discarded). Sampling becomes multi-label-aware: draw a target count per class such that every class reaches the minimum-n floor, allowing images to count toward multiple classes.
- **Per-rule test assignment:** an image is tested against **every rule it is labeled for** (not one). Compliant images are assigned rules via context-aware matching where feasible (see 3.3), else round-robin as a fallback.
- **Config:** `labeling ∈ {priority, multilabel}`; default `priority`, scale run uses `multilabel`.
- **Expected effect:** `struck_by_risk` usable pool rises from 13 to ~70 across the full 10k; enables genuine multi-hazard completeness tests.

### 3.3 Compliant-sample context matching *(reduces vacuous tests)*
- **Problem:** a compliant flat-ground image round-robined to rule_2 (harness/height) tests nothing — the hazard context doesn't exist.
- **Fix:** when assigning a rule to a compliant image, prefer a rule whose **context is present** in the scene (e.g. only assign rule_2/rule_3 to images with a height/edge context signal; only assign rule_4 where an excavator is present). Use dataset metadata fields where available; fall back to a lightweight grounding pre-check. Where no contextual match exists, mark the assignment `context_absent` so it can be reported separately rather than silently inflating baseline accuracy.
- **Config:** `compliant_assignment ∈ {round_robin, context_matched}`; default `round_robin`.

### 3.4 Worker-loss instrumentation baked into the metrics *(de-contaminates 3 metrics)*
- **Problem:** masking or perturbing the worker box can make the worker undetectable, which reroutes the proxy into a degenerate scene-level fallback and flips the answer for a reason unrelated to the safety concept. This inflates descriptive accuracy, robustness, and completeness.
- **Fix:** every metric that reruns inference after a mask/perturbation must **record post-operation `worker_boxes`** and emit a `flip_due_to_worker_loss` boolean. Report each metric twice: **raw** (all flips) and **genuine** (excluding worker-loss flips). This promotes the pilot's ad-hoc 159-rerun correction into standard metric output.
- **Config:** `report_worker_loss_corrected = true` for the scale run.

### 3.5 Prompt dictionary fix *(rule_2 concept split)*
- **Problem:** rule_2 groups "safety harness" and "fall-protection lanyard" as synonyms; they are two distinct physical objects, so a rewording test spuriously grounds different boxes.
- **Fix:** restructure the query dictionary so each **distinct physical object** has its own entry. True synonyms (guardrail / edge protection barrier) stay grouped; distinct objects (harness / lanyard) become separate concepts, each with its own compliance semantics. The rewording-robustness test only ever swaps **validated synonyms**.
- **Deliverable:** a reviewed query dictionary with, for each rule, an explicit `{concept: [synonym, ...]}` structure and a note on whether presence of any one concept, or all, satisfies the rule.

### 3.6 Unify decoding *(removes the three-regime fragmentation)*
- **Problem:** baseline/masking use beam search (`num_beams=3`), attribution uses greedy (`num_beams=1`), stability uses sampling — so metrics evaluate three different execution paths.
- **Fix:** adopt **greedy decoding (`num_beams=1`) as the single canonical path** for baseline, descriptive accuracy, sparsity, robustness, and the anchor of stability. Rationale: (a) it is the path attribution can be extracted from without beam-reordering bookkeeping; (b) efficiency profiling showed beam≈greedy cost for short grounding outputs, so nothing is lost. Stability's stochastic reruns remain sampling-based (that is the metric's point) but are compared against the greedy anchor.
- **Config:** `decoding = greedy` globally; `stability.sampling_temperature` retained as a separate parameter (see Phase 3).
- **Caveat to record:** if D4 adopts dense captioning (long outputs), re-evaluate beam-vs-greedy cost — the decode loop stops being free.

### 3.7 Hardcoded-assumption audit
- Sweep every analysis/plotting script for literal sample counts, class sizes, grid dimensions, patch-grid arithmetic (e.g. the 768→24 image-token math), and model IDs. Replace each with a value read from data or config. Add an automated check that fails CI if a known-magic literal reappears.

---

## 4. Phase 2 — Metric-validity fixes (make each metric measure what it claims)

Each of the six metrics needs an explicit control for an artifact the tabular original never faced. All additive; none replaces the pilot metric.

### 4.1 Descriptive Accuracy
- Add **text-token ablation** (the missing multimodal half the proposal promised): in addition to masking the visual region, remove the key query token(s) and re-measure the answer flip. Report visual-ablation and text-ablation flip rates **separately**, never merged.
- Add a **graded score**: expose a continuous proxy signal (fraction of workers covered / minimum distance ratio) so masking effects can be measured as magnitude, not just a boolean flip — this recovers statistical power and smooths the non-monotonicity anomaly.
- Add a **monotonic-masking invariant**: flag any sample where masking more regions un-flips an answer (the fallback-interaction signature) for audit.
- Mask-mode sensitivity: run a subset under `black`, `blur`, and `inpaint` masking; confirm flip rate is robust to mask mode (black patches are out-of-distribution artifacts).

### 4.2 Visual Sparsity
- Replace/augment the top-k-of-576 concentration measure with a **size-invariant** metric: attention mass **inside the grounded box** ÷ the box's cell footprint (an excavator covering 40% of the frame should have its attention judged against 40% of cells, not against all 576). This removes the −0.70 size confound that makes large-object rules look artificially unfocused.
- Fix and document one normalization convention (mass ratio on raw softmax attention; threshold count on min-max-normalized) and keep it consistent across the run.
- Reduce over-averaging: evaluate head/layer selection (informative heads, or last decoder layer) and per-generated-token attribution instead of averaging over all layers/heads/steps; pick the variant with the best signal on a validation subset and freeze it.

### 4.3 Stability
- Report **both** overlap scores by rule: the ranked top-region overlap and the rule's true object-box overlap.
- Add a **size-invariant** stability measure (normalized centroid-distance and/or Dice) alongside IoU, so small PPE boxes aren't penalized purely by IoU's size bias.
- Add an **object-presence-stability** metric: fraction of reruns in which the object is detected at all. (The pilot silently dropped samples where the box appeared/disappeared across reruns — the worst instability — by returning NaN; that must count as instability, not vanish.)
- **Temperature:** replace the single arbitrary `temperature=0.7` with a **sweep** (e.g. {0.3, 0.5, 0.7, 1.0}) and report a stability-vs-temperature curve. Increase reruns per sample from 3 to **≥ 10** for smoother per-sample estimates.

### 4.4 Robustness
- **Severity sweeps (dose-response):** each perturbation gets multiple magnitudes, not one fixed point. Recommended grids: Gaussian blur kernel {4, 8, 16}; gamma {1.5, 2.5, 4.0}; occlusion area-fraction {0.1, 0.2, 0.35} at both centered and random locations; contrast factor {0.15, 0.3, 0.5}. Report answer-level and explanation-level change as curves vs severity.
- **Two-number reporting** (keep from pilot): answer-level robustness and explanation-level (box-drift) robustness, never blended.
- **Verify the centerpiece finding under a size control:** the pilot's headline "28.3% of stable answers hide a drifted explanation" uses IoU, which is size-biased — bucket by object-box area and confirm the drift survives within size bands; add a size-invariant drift measure. This must be checked before the paper leans on the number.
- Report **drift and box-disappearance jointly** (don't let vanished boxes drop out of the drift denominator).
- Add **attention-heatmap drift** (from Phase 4) as a second explanation-level signal beyond box IoU.
- **Confidence proxy:** retain only as an explicitly-labeled *uncalibrated* diagnostic; never report as a correctness/trust signal (it rose under the most disruptive perturbations).
- **Occlusion:** separate *targeted* occlusion (cover the actual object box) from *random-location* occlusion, since the pilot's centered square conflated "covered the object" with "disrupted whole-scene grounding."

### 4.5 Bounded Completeness
- Bake the **worker-loss correction** (3.4) into the headline number.
- Report completeness with an explicit note that it shares construction with descriptive accuracy (they are not independent confirmations).
- **Multi-hazard corner cases** (the proposal's core promise): with multi-label data (3.2), deliberately stratify and oversample images that violate two or more rules; test whether the explanation is complete enough to capture **both** hazards (e.g. missing hard hat AND unprotected edge). Report multi-hazard completeness as its own subsection with adequate n.
- Optionally extend beyond top-1/top-2 ablation toward a higher-order necessity check (a region necessary only in combination).

### 4.6 Efficiency
- Instrument **real per-sample timing** into every inference-bearing script (retire the 15-sample calibration).
- Report **GPU-compute time separately from end-to-end wall-clock** (prove GPU-bound with local caching).
- Standardize the timing boundary (generate-only, with explicit device synchronization); discard warm-up iterations.
- Re-benchmark on the **target cluster hardware with batching**, reporting measured throughput at multiple scales (1k / 2.5k / 10k) to expose memory fragmentation / thermal effects — not a single linear extrapolation.
- Re-cost after D1 (gradient attribution) and D4 (dense captioning), both of which materially change the budget.

---

## 5. Phase 3 — Reproducibility, environment, and statistics hardening

- **Local data cache** (already implemented): the loader reads the locally-downloaded Parquet copy in preference to Hub streaming, keeping cluster runs GPU-bound and immune to the streaming `IncompleteRead` failure class.
- **Environment:** containerize (Docker/Singularity) for the cluster; pin the full dependency set. Replace the fragile `transformers` version pin with a config-wrapper that injects the missing generation attributes, so the stack survives upstream updates. Document the exact install order (pinned-CUDA torch first) to avoid the CPU-torch replacement trap.
- **Precision:** run forward-pass metrics in fp16 (fine); run **gradient-based attribution (if D1 adopts it) in bf16 or fp32** and verify the attribution rankings don't change vs fp16.
- **Attention backend:** document that SDPA on the cluster may route to FlashAttention, whose backward pass is non-deterministic — record baseline gradient matrices, and if strict cross-platform reproducibility of attribution is required, pin the attention backend.
- **Seeds:** run the full study across **multiple seeds** (recommend ≥ 5) for sample selection and any stochastic step; report mean ± CI, not a single SEED=42 point.
- **Statistics:** attach bootstrap CIs and Wilcoxon signed-rank tests to every headline metric; enforce the minimum-n floor before reporting a rate as stable.
- **Report generation:** all tables/figures in the paper and memos are generated from the results CSVs at build time (no hand-copied numbers).

---

## 6. Phase 4 — Attribution stream(s)

- **Primary:** decoder→encoder cross-attention (as in the pilot), with the verified architecture handling (drop the global pooled token before reshaping the 576 image-patch tokens into the 24×24 grid; keep this behind an assertion and recompute the patch-grid arithmetic from the actual input resolution rather than hardcoding).
- **Second stream [D1]:** Integrated Gradients on the vision encoder via Captum (already a project dependency), run in bf16/fp32. Report **agreement between cross-attention and IG** — this converts the attribution choice from a single fallback method into a corroborated result, which is what a top-tier venue expects.
- **Ranking use:** feed cross-attention/IG mass as the importance signal for `region_ranking = attention` (Phase 1.1 tier 2), giving a defensible importance ranking rather than area.
- **Note for native-VQA models (Qwen2.5-VL):** these are decoder-only, with no encoder-decoder cross-attention layer to read out. Attribution for them must use raw self-attention from selected layers/heads or a gradient method (IG) — plan for a **model-specific attribution adapter**, not a single shared method.

---

## 7. Phase 5 — Model and dataset scope [D2]

### 7.1 Models (three, with distinct jobs)
1. **Florence-2-base-ft** — keep as the instrumentable, region-native baseline. Runs the grounding-proxy pipeline.
2. **Qwen2.5-VL (3B and 7B)** — native-VQA + native grounding. Run **two explicitly-labeled comparisons**, because comparing a native-VQA model to Florence-2 on the same numbers is otherwise a confound:
   - **Same-proxy comparison:** constrain Qwen2.5-VL to the identical detect-then-threshold pipeline (it supports open-vocabulary grounding). Isolates "does a better detector make the existing proxy's explanations more faithful."
   - **Native-VQA comparison:** let Qwen2.5-VL answer each rule directly as free-text VQA, with its own attention/gradient attribution. Isolates "does end-to-end reasoning produce better explanations than a geometric proxy." **This is the comparison that makes the metrics measure model faithfulness rather than proxy sensitivity.**
   - 7B likely needs the cluster GPU and possibly quantization; 3B for faster iteration.
3. **One frontier model (e.g. GPT-4o)** — benchmark-only reference for answer accuracy. No gradient/attention attribution (API-only); include because reviewers in this space now expect a frontier comparison point.

### 7.2 Datasets (three, with distinct jobs)
1. **ConstructionSite** — scale from 163 to the full **3,004-image test split** (train split available for supplementing minority classes). Pair with the multi-label fix (3.2) so `struck_by_risk` is powered rather than inherited at n=13.
2. **SODA** (published, ~19,846 images, 15 classes, multi-site/weather/viewpoint) — **out-of-distribution robustness/completeness**: does the explanation degrade gracefully on a different site distribution, not just on perturbed versions of the same images. Requires a class/rule mapping from SODA's taxonomy to the four safety rules.
3. **CMA** (published, ~1,595 clips, 7 action classes) — **temporal explanation-stability**: extend stability from "survives stochastic reruns on one static image" to "stays consistent frame-to-frame on the same hazard across a video clip." This is the genuinely novel contribution — no current construction-VLM work does explanation-stability over time. Requires frame sampling, per-frame grounding, and a frame-to-frame explanation-overlap metric.

### 7.3 Adversarial (Level-3) robustness [D5]
- Replace the 20-image synthetic-patch stretch test with a **powered study**: realistic hard-hat patches (not a yellow ellipse), **pose-estimation-based placement** (not a geometric head-box guess), and adequate n. Report spoofing rate only over candidates for which "fooling" is well-defined (baseline already "violation"), and report placement→flip correlation with sufficient power.

---

## 8. Recommended execution order and gating

1. **Phase 1** (foundational fixes) — do first, behind flags, validate each gate. Cheap, unblocks everything.
2. **Phase 0 decisions** — resolve with Prof. Abdallah **in parallel** with Phase 1 (they gate Phases 4–5).
3. **Phase 2** (metric validity) — after Phase 1 lands and ranking is validated.
4. **Phase 3** (reproducibility/stats) — alongside Phase 2; required before any headline number is quoted.
5. **Phase 4** (attribution) — after D1; feeds Phase 1.1's attention-based ranking and Phase 2's sparsity/drift.
6. **Phase 5** (model/dataset scope) — after D2/D3/D4; the largest effort, gated on the foundations being correct so scale doesn't just produce a larger biased sample.

**Global gate before scaling sample count:** do not run the full 3,004-image study until Phase 1 fixes are validated. Scaling a biased pipeline only yields a larger, equally-biased result.

---

## 9. Deliverables checklist (per phase)

- Phase 1: flagged, validated fixes + before/after comparison on the 163-sample set showing the ranking gap narrowing and worker-loss correction applied.
- Phase 2: each metric emitting raw + corrected + size-invariant variants; text-ablation added to descriptive accuracy.
- Phase 3: containerized environment, multi-seed run harness, CI hardcoded-literal check, programmatic report generation.
- Phase 4: two attribution streams with an agreement report.
- Phase 5: three-model × three-dataset result matrix, with same-proxy and native-VQA comparisons clearly separated, and powered adversarial + temporal-stability studies.

---

## 10. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Rule-aware ranking changes recorded pilot numbers | High (intended) | Keep behind flag; freeze `area` default; report before/after explicitly |
| Dense captioning explodes decode cost | Medium | Measure on validation subset before adopting (D4); re-cost efficiency |
| Gradient attribution underflows in fp16 | High if D1 adopts it | Run attribution in bf16/fp32; verify rankings unchanged |
| SODA/CMA taxonomy doesn't map cleanly to 4 rules | Medium | Build and review an explicit mapping before running; report unmapped classes |
| Cluster FlashAttention non-determinism perturbs attribution | Low-Medium | Pin attention backend for attribution runs; record baseline matrices |
| Native-VQA vs proxy comparison read as unfair | Medium | Always run and label both same-proxy and native-VQA comparisons |
| struck_by_risk still underpowered after multi-label | Low-Medium | Supplement from train split and/or CMA; enforce minimum-n before reporting |
