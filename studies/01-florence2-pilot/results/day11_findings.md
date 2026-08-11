# Day 11 — Analysis & Figures Rollup

Pure rollup of Days 5-10's already-computed CSVs into one summary table and one chart —
no new model calls, no recomputation. `results/pilot_metric_summary.csv` (6 rows, one per
metric), `results/figures/summary/metric_summary_by_class.png`. As with Days 1-10, visual
review was done by reading saved PNGs directly rather than via a Jupyter notebook
(`notebooks/` was never actually used in this pilot — every day's visual check happened
through Claude Code's `Read` tool on saved figures, which renders images directly).

## Headline table

| metric | overall | compliant (n=50) | ppe_violation (n=50) | fall_hazard (n=50) | struck_by_risk (n=13) | source |
|---|---|---|---|---|---|---|
| descriptive_accuracy (answer flips when top-1 region masked) | 36.2% | 28.0% | 28.0% | 46.0% | 61.5% | `descriptive_accuracy.csv` |
| visual_sparsity_top5 (top-5-token attention mass ratio) | 0.108 | 0.099 | 0.127 | 0.109 | 0.067 | `visual_sparsity.csv` |
| stability (3-run sampled-answer agreement) | 77.5% | 74.7% | 76.0% | 78.7% | 89.7% | `stability.csv` |
| robustness Level 1 (answer survives blur/low_light/occlude/contrast_shift) | 80.7% | 79.5% | 86.0% | 75.0% | 86.5% | `robustness.csv` |
| bounded_completeness (`explanation_supported` rate) | 43.6% | 32.0% | 36.0% | 58.0% | 61.5% | `bounded_completeness.csv` |
| efficiency, full 6-step pipeline extrapolated to n=1000 | ~0.50 GPU-hours (~30 min), single RTX 3070 | n/a (aggregate runtime budget, not per-sample) | | | | `efficiency_summary.csv` |

**struck_by_risk has only 13 samples** (the balanced 50-per-class sampler couldn't find
50 in the 3,004-image test split — already noted in Day 2). Every struck_by_risk number
above rests on n=13, not n=50; read it as a directional signal, not a stable estimate.

## Finding 1: descriptive_accuracy and bounded_completeness move together by class — but that's expected, not independent confirmation

Both metrics are weakest for compliant (28.0% / 32.0%) and ppe_violation (28.0% / 36.0%),
and strongest for fall_hazard (46.0% / 58.0%) and struck_by_risk (61.5% / 61.5%). This
isn't two metrics independently agreeing — Day 10's `bounded_completeness` is *built from*
Day 5's `answer_changed_top1`/`top2` columns by construction, so the two trends sharing a
shape is close to a methodological identity, not a second confirmation. The real,
independent explanation for *why* both are low for PPE/compliant was already traced in
Day 10 Finding 1: `standardize_regions`' area-based ranking promotes the worker's
whole-body box over the smaller hard-hat/vest box for rule_1, and masking the worker is
less likely to flip the proxy's answer than masking the actual safety object (37.5% vs.
49.4% supported, Day 10). This cross-metric view doesn't add new evidence for that claim;
it just makes the consequence visible in one chart.

## Finding 2: visual sparsity runs in the opposite direction from descriptive_accuracy/completeness for struck_by_risk

struck_by_risk has the *highest* descriptive_accuracy (61.5%) and completeness (61.5%) of
any class, but the *lowest* visual sparsity (0.067 vs. 0.099-0.127 elsewhere) — its
explanation is the least concentrated, yet the most load-bearing by the masking test.
This is a real, if weak (n=13, 4-class comparison), signal that "focused attention" and
"the masked region was decision-critical" are not the same property and can point in
different directions. A plausible mechanism: rule_4's attributed objects (excavator,
worker) are large structural regions, not small point-like objects like a hard hat —
Florence-2's grounding-query attention naturally spreads over a bigger area for a bigger
object, lowering top-5-token mass concentration, even though masking that whole region
still reliably destroys the spatial worker-excavator relationship the proximity check
depends on. Consistent with the pilot's running theme (Day 6/9): sparsity and
accuracy/completeness measure genuinely different things and must be reported
separately, not blended into one "explanation quality" score.

## Finding 3: robustness is the flattest metric across classes, and fall_hazard is its worst case (not its best, like elsewhere)

Robustness Level 1 (75.0%-86.5%) varies far less across classes than the other four
metrics (which all span roughly 30-60 percentage points). fall_hazard is the single worst
class here (75.0%), even though it's the second-best class for descriptive_accuracy and
completeness. This matches, rather than contradicts, what Day 9 already found: fall_hazard
pools rule_2 and rule_3 samples, and Day 9 showed these two sub-rules destabilize for two
different reasons — rule_2 has the highest `worker_lost` fallback-rerouting rate under
occlusion (19.2%, Day 9 Finding 1), while rule_3 has the highest grid-fallback rate (Day
10 Finding 3, 3/4 grid-fallback samples are rule_3). Averaging two different failure modes
into one "fall_hazard" number here makes the class look moderately fragile without
revealing either underlying cause — both are already documented at the rule level in
their respective days' findings.

## Finding 4: efficiency comfortably scales — ~30 minutes of GPU time for the full 6-metric pipeline at n=1000

Summing Day 8's `extrapolated_1000_hours` across all six logged pipeline steps
(baseline inference, descriptive-accuracy reruns, attribution, stability's 3x sampled
reruns, robustness's perturbation reruns, completeness's top-1-mask reruns) gives ~0.50
GPU-hours, well under an hour on a single RTX 3070. This is the one metric where bigger
is unambiguously better news for a future full-dataset study — there's no compute
bottleneck standing between this pilot and a 1000+ sample follow-up.

## Verification performed

- `scripts/11_make_figures.py` ran end-to-end, exit code 0, wrote 6 rows to `pilot_metric_summary.csv` and the chart.
- Every number in `pilot_metric_summary.csv` traces directly to one of Days 5-10's already-verified CSVs by a single `groupby(primary_class).mean()` — no new model calls, no recomputed logic, consistent with the plan's stated Day 11 scope.
- Visually inspected `figures/summary/metric_summary_by_class.png` (read directly) to confirm the bar heights match the printed table before writing the findings above off the chart.
- Cross-checked the descriptive_accuracy/completeness non-independence note against `bounded_completeness.csv`'s actual classification logic (`metrics/completeness.py`) rather than assuming agreement meant two separate confirmations.
