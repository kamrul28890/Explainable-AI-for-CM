# Day 5 Findings: Descriptive Accuracy

Ran `scripts/05_descriptive_accuracy.py` over all 163 pilot samples: masks
each sample's top-1 standardized region (Day 4), reruns `answer_rule`, and
checks whether the answer flips from baseline; then masks top-1+top-2
together and checks again. Output: `results/descriptive_accuracy.csv`, 10
flip-example overlays in `results/figures/descriptive_accuracy/`.

## Results

| Masking level | Descriptive accuracy (answer flip rate) |
|---|---|
| Top-1 region | **36.2%** |
| Top-1 + top-2 regions | 35.0% |

By `primary_class` (top-1): `struck_by_risk` 61.5%, `fall_hazard` 46.0%,
`ppe_violation` 28.0%, `compliant` 28.0%.

By `assigned_rule_id` (top-1): rule_4 44.0%, rule_3 42.9%, rule_2 38.5%,
rule_1 27.0%.

These numbers track Day 3's redesigned-proxy sensitivity/specificity
ordering closely: rule_4 and rule_3 (which already had the strongest real
discriminative signal) also have the highest descriptive accuracy, while
rule_1 (weakest signal, 16% sensitivity) also flips least often when its
top region is masked -- consistent, not coincidental: a proxy that barely
discriminates also has little for masking to disrupt.

## Finding: masking top-2 is not strictly stronger than masking top-1

Naively, masking *more* of the evidence should only ever flip the answer as
often or more often than masking less -- it should never *un-flip* a result
that top-1 alone already flipped. Here it does, in 15/163 samples (top-2 accuracy
35.0% < top-1's 36.2%). Traced one concretely (image `0000069`, rule_1):

- Baseline boxes: one worker box (area 10,778 px²) and one hard-hat box
  (area 891 px²). The worker box is *larger*, so Day 4's area-based ranking
  puts it at top-1, ahead of the hard-hat box.
- **Masking top-1 (the worker)**: the rerun can no longer detect that worker
  at all -- `worker_boxes` comes back empty. `_answer_presence_rule` falls
  back to its scene-level branch (see `inference.py`): "no worker detected,
  so answer compliant if the object exists anywhere." The hard-hat box is
  still visible (only the worker was masked), so the fallback answers
  **compliant** -- a flip from baseline's `violation`.
- **Also masking top-2 (the hard hat)**: now neither the worker nor the
  object is visible. The same fallback branch now finds no object either,
  so it answers **violation** -- back to the original baseline answer.

This is a real interaction between the masking test and the person-relative
redesign's own fallback logic (Day 3), not a bug in either: masking the
*worker* region doesn't just remove "evidence," it removes the precondition
(`if not worker_boxes`) that the per-worker check needs to run at all,
silently rerouting to the cruder scene-level branch. That branch's answer
then depends only on whether the *other* region is still visible -- which
can flip the result in either direction independent of how much total area
is masked. 14 of the 15 regressions follow this exact pattern (`masked_answer_top2
== baseline_answer` -- full revert, not partial); confirmed visually for
`0000069` (`results/figures/descriptive_accuracy/0000069_rule_1_flip.png`
shows the masked box is the worker, not the hard hat).

## Implication for Days 9-10

Robustness (Day 9) and Bounded Completeness (Day 10) both rerun `answer_rule`
on modified images and compare to baseline, the same shape of test as this
day. The same fallback-rerouting behavior will apply there: a perturbation
or mask that happens to remove the only detected worker (rather than the
safety object) can flip the answer for reasons disconnected from the
*safety concept* the rule is meant to test. Worth checking for this pattern
specifically in those days' results rather than only reporting the
aggregate flip rate, since it's now a known, recurring shape of false
positive/negative for this proxy family.

## Visual confirmation

- `0000069_rule_1_flip.png`: top-1 region is the worker box (small, around
  a man in a red hard hat in the foreground) -- masking it removes that
  worker from detection entirely, confirmed visually.
- `0000120_rule_4_flip.png`: top-1 region is the excavator box itself --
  masking it removes the proximity check's other half, flipping `hazard`
  -> `safe`. This one *is* the intuitive case: the masked region is exactly
  the rule-defining object, and removing it removes the hazard signal in a
  way that matches what "explanation" should mean.
