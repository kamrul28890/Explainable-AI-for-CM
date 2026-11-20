# Full-Pipeline Walkthrough Notebook — Design

**Date:** 2026-08-06
**Status:** approved

## Goal

A single executable notebook that shows the entire ConstructionSite XAI pipeline
end to end, with a visualization at every step, so a reader can understand what
the pipeline does without reading eleven scripts.

## Contract

- **New file:** `pilot/notebooks/00_full_pipeline_walkthrough.ipynb`. The existing
  stub `constructionsite_xai_pipeline_walkthrough.ipynb` is left untouched.
- **No training.** Florence-2-base-ft loads pre-trained and runs zero-shot, which
  is what the pipeline does anyway.
- **Live GPU execution** on a small subset (`N_PER_CLASS = 4`, 16 samples),
  committed with outputs baked in.
- **Frozen-pilot config.** `REGION_RANKING="area"`, `LABELING="priority"`,
  `COMPLIANT_ASSIGNMENT="round_robin"`, `DECODING="beam"`,
  `REPORT_WORKER_LOSS_CORRECTED=False`. Asserted in section 1 so the notebook
  fails loudly if a default drifts.
- **Read-only w.r.t. frozen artifacts.** Writes nothing to `pilot/results/*.csv`.
  Figures go to `pilot/results/figures/notebook/`.
- **Audience:** Prof. Abdallah and reviewers. Assumes XAI framework knowledge;
  explains the construction-VLM adaptation and the proxy-bias caveats.

## Structure

Sixteen sections. Every code cell is followed by a visual or a table.

| § | Content | Visual |
|---|---|---|
| 0 | Scope, contract, what this is not | config table |
| 1 | Environment + version check, config flags asserted | flag table |
| 2 | Dataset load, schema | class distribution bar, rule co-occurrence heatmap |
| 3 | Balanced sample selection, live | manifest head, overlap vs frozen `pilot_samples.csv` |
| 4 | Grounding-as-VQA (Design Decision #1) | `RULE_QUERIES` table, proxy flow |
| 5 | Baseline inference, one image, step by step | worker boxes → object boxes → proximity → verdict, 4-panel |
| 6 | Baseline over the subset | confusion matrix, per-class accuracy, confidence histogram |
| 7 | Region standardization | ranked-region overlay, area-ranking bias demo |
| 8 | Descriptive Accuracy | original/top1/top2 masked panels, flip table, rate ± CI vs frozen |
| 9 | Sparsity | cross-attention overlay, Gini + top-k mass curve, box mass |
| 10 | Stability | sampled reruns, agreement rate, centroid scatter |
| 11 | Efficiency | per-stage timing bar, extrapolation to n=1000 |
| 12 | Robustness | 4-perturbation grid, survival rate per perturbation |
| 13 | Bounded Completeness | verdict stacked bar |
| 14 | Roll-up: subset vs frozen 163 | side-by-side bars with CIs |
| 15 | Limitations and what Phases 1–3 fixed | — |

## Two explicit hazards, surfaced in the notebook

1. **n=16 is below `stats.MIN_N_FLOOR = 30`.** Bootstrap CIs are computed but
   every subset number carries a visible "below reporting floor — illustrative
   only" banner. The frozen 163-sample numbers remain the authoritative ones.
   Prevents a reader quoting a notebook number as a result.

2. **Section 9 uses greedy decoding** while 5–8 use beam. This is the pilot's
   known decoding fragmentation (`attribution.py` module docstring: beam search
   reorders the batch dimension, so cross-attentions do not line up). Stated as a
   visible note rather than silently mixing regimes.

## API surface used

Existing library only; no new library code.

- `data.load_construction_site`, `data.select_balanced_sample`, `data.classify_image`
- `model.load_florence2`, `model.run_task`
- `inference.answer_rule`, `inference.AnswerResult`
- `regions.standardize_regions`, `regions.mask_region`, `regions.iou`
- `prompts.RULE_QUERIES`, `prompts.RULE_OBJECT_LABEL`, `prompts.rule_object_labels`
- `attribution.cross_attention_heatmap`
- `metrics.descriptive_accuracy.evaluate`
- `metrics.sparsity.{topk_mass_ratio, gini_concentration, box_mass_inside, regions_above_threshold}`
- `metrics.stability.{run_n_times, answer_agreement_rate, object_presence_rate, mean_pairwise_centroid_distance}`
- `metrics.efficiency.{time_call_cuda, summarize_timings, extrapolate}`
- `metrics.robustness.evaluate`
- `metrics.completeness.classify_sample`
- `perturbations.{blur, low_light, occlude, contrast_shift}`
- `stats.{bootstrap_ci, meets_min_n}`
- `viz.{overlay_boxes, overlay_heatmap}`

## Verification

The notebook is executed end to end with the pilot venv kernel before commit.
Execution succeeding *is* the test: every cell runs against the real library and
real GPU, so an API drift or a shape error fails the build.
