# Key Visualizations for the Meeting

Curated subset of `results/figures/` (which has 60+ saved images) — these are the ones worth pulling up live. Each is tied to a specific finding in `pilot_report_draft.md`, not picked for being pretty.

## 1. The headline rollup

`results/figures/summary/metric_summary_by_class.png`
All six metrics, broken down by hazard class. Start here — everything else explains *why* this chart looks the way it does.

## 2. The grounding-proxy adaptation, working

`results/figures/baseline/0000007_rule_1.png`
A baseline grounding overlay (worker box + hard-hat box) — what the redesigned person-relative proxy actually produces, the thing that went from 3.0% to 27.0% sensitivity (Day 3).

## 3. What "descriptive accuracy" looks like on a real sample

`results/figures/descriptive_accuracy/0000037_rule_1_flip.png`
Masking the top-ranked region flips the answer — the clearest single-image illustration of the metric in Section 5.

## 4. Sparsity: focused vs. scattered, same metric, two physically different targets

`results/figures/sparsity/0000023_rule_1_hard_hat.png` (focused — small, sharp hotspot on a distant hard hat)
`results/figures/sparsity/0000079_rule_3_guardrail.png` (scattered — attention spread across most of the frame for a guardrail spanning the image)
Side by side, these make Section 6.2/6.4's point without needing the correlation number: sparsity partly reflects target size, not just explanation quality (Day 6).

## 5. A stable answer hiding a drifted explanation

`results/figures/robustness/verify/0000034_before.png` → `results/figures/robustness/verify/0000034_after_occlude.png`
Same final answer before and after occlusion; the model's evidence box quietly moves to something unrelated (`object_box_iou=0.005`). The single best image for Section 6.3's claim that answer-level robustness alone is not enough (Day 9).

## 6. The patch-stretch test's counterintuitive result

`results/figures/robustness/verify/0000641_patch_before.png` / `_after.png` — on-target placement, **did not** flip
`results/figures/robustness/verify/0000233_patch_before.png` / `_after.png` — off-target placement, **did** flip
Useful for the "what failed, and why we're not overclaiming it" part of the talking points — n=4 is too small to generalize, and this pair is exactly why (Day 9 stretch task).

## 7. Compute is not the obstacle

`results/figures/efficiency/per_sample_cost.png`
Per-sample timing breakdown behind the "~0.5 GPU-hours at n=1000" number (Day 8).
