# Two-Week Pilot Plan for the Construction VLM XAI Proposal

## Can We Actually Do It in Two Weeks?

Yes, we can complete a credible two-week pilot, but only if the scope stays bounded.

The two-week goal is not to implement the full research proposal. The goal is to prove that Professor Abdallah's six-metric XAI evaluation framework can be adapted to a small construction VLM setting.

## Realistic Two-Week Scope

We can realistically finish:

- Use ConstructionSite 10k only for the pilot.
- Use Florence-2 as the first model.
- Use 100 to 300 samples, not the full dataset.
- Run safety-rule prompts.
- Generate predictions and grounding outputs.
- Test explainability through region masking and occlusion.
- Add limited attention or gradient attribution only if technically stable.
- Compute a small set of metrics:
  - descriptive accuracy,
  - visual sparsity,
  - stability,
  - efficiency,
  - small robustness test,
  - bounded completeness.

We should not promise in two weeks:

- full LRP or DeepLIFT on Florence-2,
- full CMA temporal/video integration,
- full SODA benchmark,
- full adversarial model retraining,
- full mathematical completeness over all samples.

## Day-by-Day Task Breakdown

## Day 1: Environment and Dataset Access

Goal: confirm the pilot is executable.

Tasks:

- Set up the Python environment.
- Install required packages:
  - `torch`,
  - `transformers`,
  - `datasets`,
  - `Pillow`,
  - `opencv-python`,
  - `numpy`,
  - `pandas`,
  - `matplotlib`.
- Verify that Florence-2 loads locally.
- Verify access to ConstructionSite 10k.
- Inspect dataset fields:
  - images,
  - captions,
  - VQA,
  - boxes,
  - labels,
  - image attributes.

Deliverable:

- Environment works.
- Dataset sample loads.
- Florence-2 runs on one image.

## Day 2: Sample Selection

Goal: create the pilot subset.

Tasks:

- Select 100 to 300 samples.
- Balance samples across:
  - compliant or safe scenes,
  - PPE violations,
  - fall hazards,
  - struck-by or proximity risks.
- Save sample metadata to CSV or JSON.
- Define 5 to 10 safety-rule prompts.

Deliverable:

- `pilot_samples.csv`
- `safety_prompts.json`

## Day 3: Baseline Florence-2 Inference

Goal: get baseline model outputs.

Tasks:

- Run Florence-2 on all pilot samples.
- Store:
  - generated answer,
  - caption if available,
  - grounding output,
  - bounding regions,
  - runtime per sample.
- Manually inspect 10 to 20 outputs for sanity.

Deliverable:

- `baseline_predictions.csv`
- sample visualization folder.

## Day 4: Grounding and Region Extraction

Goal: convert outputs into explainable visual units.

Tasks:

- Extract bounding boxes or grounded regions.
- If boxes are missing, use image patches or simple grid regions.
- Standardize each image into candidate visual regions.
- Create functions for:
  - masking a region,
  - blurring a region,
  - cropping or highlighting a region,
  - saving a visualization.

Deliverable:

- Region extraction pipeline.
- Example before-and-after masked images.

## Day 5: Descriptive Accuracy Test

Goal: test whether highlighted regions matter.

Tasks:

- For each sample, mask the top-1 region.
- Re-run Florence-2.
- Compare original answer with masked answer.
- Compute answer-change rate or confidence proxy change.
- If feasible, also test top-2 regions.

Deliverable:

- Descriptive accuracy results.
- Plots or tables showing answer changes after masking.

## Day 6: Visual Sparsity Metric

Goal: measure whether explanations are focused.

Tasks:

- Estimate attribution concentration over regions.
- Compute simple visual sparsity:
  - number of regions above a threshold,
  - percentage of attribution mass in top-k regions,
  - concentration ratio.
- Compare good and poor examples.

Deliverable:

- Visual sparsity table.
- 5 to 10 example heatmaps or region overlays.

## Day 7: Stability Test

Goal: test repeatability.

Tasks:

- Re-run the same prompts three times on the same sample subset.
- Compare selected regions across runs.
- Compute overlap using:
  - IoU if boxes are available,
  - top-k region overlap if using grids,
  - answer consistency.

Deliverable:

- Stability score table.
- Examples of stable and unstable explanations.

## Day 8: Efficiency Test

Goal: measure practical runtime.

Tasks:

- Record:
  - inference time,
  - masking time,
  - re-inference time,
  - explanation time,
  - memory use if possible.
- Report average, median, and max time per sample.
- Estimate scaling to 1,000 samples.

Deliverable:

- Efficiency summary.
- Runtime plot.

## Day 9: Robustness Test

Goal: test simple noise sensitivity.

Tasks:

- Apply controlled perturbations:
  - blur,
  - low light,
  - occlusion,
  - contrast shift.
- Re-run Florence-2.
- Compare:
  - prediction changes,
  - grounding region changes,
  - explanation region drift.

Deliverable:

- Robustness table.
- Visual examples showing success and failure cases.

## Day 10: Bounded Completeness Test

Goal: test whether top regions are actually necessary.

Tasks:

- For each selected sample:
  - mask the top-1 region,
  - optionally mask the top-2 region,
  - check whether the answer changes.
- Mark samples as:
  - explanation supported,
  - explanation weak,
  - no usable explanation.
- Focus on hard examples:
  - occlusion,
  - crowded scenes,
  - multiple workers,
  - subtle PPE misuse.

Deliverable:

- Bounded completeness results.
- Failure-case examples.

## Day 11: Analysis and Figures

Goal: turn raw results into professor-ready evidence.

Tasks:

- Create figures:
  - baseline examples,
  - masked-region examples,
  - robustness examples,
  - metric summary bar chart.
- Summarize results by class.
- Identify the most important failure patterns.

Deliverable:

- `figures/`
- `pilot_metric_summary.csv`

## Day 12: Draft Pilot Report

Goal: write the technical memo.

Tasks:

- Write a 4 to 6 page report covering:
  - motivation,
  - dataset,
  - model,
  - methods,
  - six metrics,
  - preliminary findings,
  - limitations,
  - next steps.
- Be explicit about what was bounded.

Deliverable:

- Pilot report draft.

## Day 13: Polish and Validate

Goal: make it presentable.

Tasks:

- Clean tables.
- Improve figures.
- Verify code runs from scratch on a small subset.
- Make a one-page professor summary.
- Prepare talking points.

Deliverable:

- Final pilot report.
- One-page meeting brief.

## Day 14: Meeting Package

Goal: prepare the professor-facing package.

Tasks:

- Finalize:
  - one-page pitch,
  - pilot report,
  - sample visualizations,
  - task plan for the full study.
- Prepare a 5-minute explanation:
  - what we tested,
  - what worked,
  - what failed,
  - where Professor Abdallah's expertise is needed.

Deliverable:

- Complete collaboration package.

## Recommended Two-Week Goal

The two-week goal should be:

> Prove that Abdallah's six-metric framework can be adapted to a small ConstructionSite 10k and Florence-2 pilot using grounded regions, masking, perturbation, and bounded explanation metrics.

This is enough to walk into a meeting and say:

> I read your framework, mapped it to construction VLMs, and built a small pilot showing how your six metrics can evaluate whether VLM safety explanations are trustworthy. The next step is to formalize the white-box attribution methods with your guidance.

## Final Recommendation

Do not try to implement the full proposal in two weeks.

Instead, build a narrow but convincing pilot:

- one dataset,
- one model,
- 100 to 300 samples,
- simple safety prompts,
- grounded visual regions,
- masking-based explanation tests,
- limited robustness,
- bounded completeness.

This is realistic, technically honest, and useful for a professor meeting.
