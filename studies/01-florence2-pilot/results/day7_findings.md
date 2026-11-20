# Day 7 Findings: Stability

## Avoiding a vacuous result (per the plan's own warning)

Florence-2's default decoding (beam search, no sampling) is deterministic --
three literal reruns would be 100% "stable" by construction and would prove
nothing. `metrics/stability.py` reruns each sample 3x with `do_sample=True,
num_beams=1, temperature=0.7` instead. Confirmed this isn't vacuous before
trusting any aggregate: mean `answer_agreement_rate` across all 163 samples
is **0.775**, not 1.0 -- roughly a quarter of rerun pairs genuinely
disagree on the final answer. The deterministic beam-search answer (already
logged in `baseline_predictions.csv`) is reported separately as
`deterministic_matches_majority`, not folded into the stability score.

## Results

Ran `scripts/07_stability_test.py` over all 163 samples: 3 sampled reruns
each, plus comparison against the existing deterministic answer.

| Metric | Mean (163 samples) |
|---|---|
| `answer_agreement_rate` (pairwise answer match across 3 reruns) | 0.775 |
| `deterministic_matches_majority` (beam-search vs. sampled-majority answer) | 83.4% |
| `top_region_overlap_score` (IoU of area-ranked top-1 region) | 0.667 |
| `object_region_overlap_score` (IoU of the rule's safety-object box specifically) | 0.602 |

By `assigned_rule_id`:

| Rule | answer_agreement_rate | object_region_overlap_score |
|---|---|---|
| rule_1 (hard hat) | 0.735 | 0.499 |
| rule_2 (harness) | 0.744 | 0.533 |
| rule_3 (guardrail) | 0.782 | 0.615 |
| rule_4 (excavator proximity) | 0.893 | 0.922 |

## Finding: `top_region_overlap_score` hides how unstable the safety-relevant box actually is

`top_region_overlap_score` (0.667 mean) looks reasonably stable, but it's
computed on Day 4's area-ranked top-1 region -- which for rules 1-3 is
almost always the **worker** box (much larger than the PPE item), not the
hard hat / harness / guardrail box the rule is actually about. Added
`object_region_overlap_score` specifically on `object_boxes[0]` (the rule's
actual safety object) to check this, and the gap is large: rule_1's hard-hat
box overlaps only **0.499** across reruns vs. rule_4's excavator box at
**0.922** -- nearly 2x more stable.

Ruled out one easy alternative explanation first: could this just be
multiple detected boxes getting reordered across stochastic runs (so
`object_boxes[0]` picks a *different* real box, not a *moved* one)? Checked
directly -- only 4-8% of samples have more than one object box at all (rule_1
6.3%, rule_2 3.8%, rule_3 8.2%, rule_4 4.0%), so reordering can't explain a
systematic ~2x gap across 24-63 samples per rule.

The real driver looks like a combination of **object size and shape**,
parallel to Day 6's box-area confound in the sparsity metric:

| Rule | Object | Median box area (px²) | object_region_overlap_score |
|---|---|---|---|
| rule_1 | hard hat | 570 | 0.499 |
| rule_2 | harness | 7,169 | 0.533 |
| rule_3 | guardrail | 221,163 | 0.615 |
| rule_4 | excavator | 215,069 | 0.922 |

Size alone doesn't fully explain it -- rule_3 (guardrail) and rule_4
(excavator) have almost identical median box area (221k vs. 215k px²) but
very different stability (0.615 vs. 0.922). The remaining gap looks like
shape/rigidity: a guardrail is a long, thin, sometimes-segmented structure,
so small differences in where the model decides the run starts or stops
shift the box's extent noticeably; an excavator is one compact, rigid,
visually distinctive object whose boundary doesn't depend much on stochastic
decoding. Hard hats and harnesses are small *and* tightly bound to a
worker's body, so a few pixels of box-edge jitter is a large fraction of the
box's own size, tanking IoU even when the model is clearly still looking at
roughly the same spot.

**Visual confirmation:**
- `results/figures/stability/0000023_rule_1_reruns.png` (hard hat,
  `object_region_overlap_score=0.069`): the overlaid top-1 boxes (worker,
  large) sit almost exactly on top of each other across all 3 reruns --
  the worker detection is stable. The hard-hat box itself (not drawn in this
  overlay, since it loses the area-ranking to the worker box) is what
  actually varies between runs.
- `results/figures/stability/0000079_rule_3_reruns.png` (guardrail,
  `object_region_overlap_score=0.955`): the formwork/guardrail wall box is
  nearly identical across all 3 reruns -- a large, well-bounded structure
  the model locates consistently regardless of sampling.

## Implication for Day 11

Both Day 6 (visual sparsity) and Day 7 (stability) show the same underlying
pattern: metrics computed on small, body-worn PPE items (hard hat, harness)
look systematically worse than metrics on large rigid objects (excavator),
for reasons that are at least partly mechanical (box size/shape), not purely
about explanation quality. Worth flagging together in Day 11's cross-metric
analysis rather than treating rule_1/rule_2's weaker numbers at face value --
and worth noting the irony: hard hat and harness are exactly the PPE items
where a *stable* explanation matters most for real-world trust, and they
are the least stable by this measure.
