# Day 8 Findings: Efficiency

## Instrumentation gap (discovered, not assumed away)

The plan called for `aggregate_timings()` to pull `inference_ms` straight out
of Days 3/5/6/7's CSVs with zero new model calls. Checking the actual
headers first (per this pilot's "verify, don't assume" habit) showed only
`baseline_predictions.csv` (Day 3) ever logged it -- `descriptive_accuracy.csv`,
`visual_sparsity.csv`, and `stability.csv` don't have a timing column at all.
Handled each gap the cheapest honest way available instead of inventing
numbers:

- **Day 5** reruns the *exact same call type* Day 3 already timed
  (`answer_rule`, default `num_beams=3`). Its cost is recovered exactly --
  no new model calls -- as `rerun_count x that sample's own inference_ms`,
  where `rerun_count` (1, or 2 if a second region existed) comes straight out
  of `region_extraction.csv`'s `n_regions` column.
- **Days 6 and 7** use two decoding configs nothing had timed before
  (`output_attentions=True`; `do_sample=True, num_beams=1`), so this script
  runs one small, fresh 15-sample calibration for each -- a deliberate, bounded
  deviation from "no new model calls," scoped the same way Day 6 scoped its
  own method deviation: small enough not to threaten the day's timebox,
  documented rather than hidden.

## Results

| Day | What | Calls/sample | Mean ms/sample (163-sample pilot) | Extrapolated to n=1000 |
|---|---|---|---|---|
| 3 | baseline inference | 1 `answer_rule` call (2 generate calls inside) | 271 | 4.5 min |
| 5 | descriptive-accuracy reruns | 1.94 `answer_rule` calls | 526 | 8.8 min |
| 6 | cross-attention attribution | 1 `cross_attention_heatmap` call (1 generate call), 15-sample calibration | 215 | 3.6 min |
| 7 | stability sampled reruns | 3 `answer_rule` calls | 778 | 13.0 min |

**Full 6-metric pipeline at n=1000 (linear extrapolation): ~0.50 GPU-hours
(~30 minutes) on a single RTX 3070.** Practically trivial -- efficiency is
not a blocker for scaling this pilot to the full 3,004-image test split, let
alone a 1,000-sample study.

(Day 4 region extraction and Day 11 analysis do no model inference at all --
correctly excluded from this table; they're pandas/PIL operations on
already-computed CSVs.)

## Finding 1: comparing "ms per call" across days needs call-count normalizing

Day 6's 215ms looks cheaper than Day 3's 271ms at face value, which seems to
contradict Day 6's own finding that `output_attentions=True` forces a slower
eager-attention path. The catch: `answer_rule` (Days 3/5/7) makes **two**
generate calls per "call" (worker phrase + object phrase), while
`cross_attention_heatmap` (Day 6) makes **one**. Normalizing to a
single-generate-call basis: Day 3/7's implied per-generate-call cost is
~271/2 ≈ 135ms and ~259/2 ≈ 130ms; Day 6's single call costs 215ms --
**about 65% more per call**, consistent with the eager-attention overhead
documented in `day6_findings.md`, once the comparison is apples-to-apples.

## Finding 2: `num_beams=3` beam search is barely slower than `num_beams=1` sampling here

Day 7's sampled calls (`num_beams=1`, 259ms/call) are only ~4% faster than
Day 3's beam-search calls (`num_beams=3`, 271ms/call) -- not the ~3x
difference a naive "3 beams = 3x the decode compute" intuition would predict.
The likely reason: these grounding outputs are very short (location-token
sequences, ~7-8 generated tokens per the Day 6 investigation), so each call's
latency is dominated by the *fixed* cost of encoding the image through the
DaViT vision tower once per call -- a cost that doesn't depend on beam count
at all -- rather than by the decode loop where beam multiplicity would
actually matter. Beam count would likely matter far more for tasks that
generate long sequences (e.g. `<MORE_DETAILED_CAPTION>`); it barely matters
for this pilot's short grounding-token outputs.

## Sanity check on the extrapolation

Day 3 and Day 5's numbers are exact sums over the real 163 samples (44.2s
and 85.7s respectively) -- no extrapolation error possible there, they're
direct aggregation. Day 6 and 7's numbers are a 15-sample calibration scaled
linearly to 163 (and then to 1000); the calibrated 778ms/sample for Day 7
implies ~127s of pure GPU compute for all 163 samples x 3 reruns, which is
the right order of magnitude for what was actually observed running
`07_stability_test.py` end-to-end (a few minutes wall-clock, most of the
difference being one-time model load and HF dataset streaming overhead, not
GPU compute) -- consistent, not off by an order of magnitude.

## Caveat for Day 11/12

Day 6 and 7's per-sample costs in this table rest on a 15-sample calibration,
not the full 163-sample run (unlike Day 3/5's exact aggregation). Good
enough for an order-of-magnitude efficiency story; should be re-measured
properly (instrument `inference_ms` directly into those scripts) before
quoting precise numbers in anything more formal than this pilot.
