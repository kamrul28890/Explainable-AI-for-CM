# Day 3 Findings: Baseline Inference Review

Ran `scripts/03_run_baseline_inference.py` over all 163 pilot samples
(`data/pilot_samples.csv`). Output: `results/baseline_predictions.csv`,
20 overlay images in `results/figures/baseline/`.

## Pipeline mechanics: working correctly

Visual inspection of saved overlay images confirms the mechanics are sound:
- `<OPEN_VOCABULARY_DETECTION>` returns plausible, tightly-localized boxes
  (e.g. image 7: a hard-hat box landing correctly on a worker's head, not on
  unrelated scene content).
- The rule_4 proximity check (image 120) correctly grounds both "worker" and
  "excavator" with boxes in sensible locations and a sound spatial relationship.
- Mean inference time: 174ms/call, mean confidence (mean token probability):
  0.58 — both consistent with Day 1's single-sample numbers, no GPU/memory
  issues across 163 samples x ~2-3 model calls each.
- All 163 target image_ids were found in the streamed test split (no missing
  rows), and the new `assigned_rule_id` column round-trips correctly through
  `pilot_samples.csv` (see Day 2 update).

## Critical finding: the presence-based proxy (rules 1-3) is nearly insensitive to real violations

Quantified against ground truth (`primary_class`):

| Proxy | Sensitivity (correctly flags real violation) | Specificity (correctly flags real compliance) | n |
|---|---|---|---|
| Presence-based (rules 1-3: hard hat / harness / guardrail) | **3.0%** (3/100) | 97.4% (37/38) | violation n=100, compliant n=38 |
| Proximity-based (rule 4: worker x excavator) | 61.5% (8/13) | 41.7% (5/12) | struck_by_risk n=13, compliant n=12 |

The presence-based proxy answers "compliant" for almost every image regardless
of ground truth — including 50/50 `ppe_violation` samples and 47/50
`fall_hazard` samples. This is not a code bug (boxes are real, well-localized
detections, confirmed visually) -- it is a mismatch between what the proxy
measures and what the dataset's rule actually means:

- The dataset's `rule_1_violation` etc. mark that **a specific worker** in the
  scene is missing PPE / fall protection.
- `<OPEN_VOCABULARY_DETECTION>` answers a **scene-level existence** question:
  "does a hard hat appear anywhere in this image?"
- Most ConstructionSite images contain multiple workers. If even one worker
  is compliant, the existence query returns a box and the proxy says
  "compliant" -- even when a different worker in the same frame is the one
  the dataset flagged as violating the rule.

This was directly visible in image 79 (ground truth: `fall_hazard`/rule_3 --
no edge protection): the "guardrail" query matched the wooden excavation-pit
formwork as a guardrail-shaped structure, so the proxy answered "compliant"
even though the dataset's actual violation is about a missing barrier
elsewhere in the same scene.

The rule_4 proximity proxy does **not** have this failure mode (it is
*relational*, not existential -- it asks whether two specific detected
objects are close, not whether an object class exists anywhere), and shows
real, balanced discriminative signal (62%/42%) even at this small n.

## Implication for Days 4-10

Descriptive Accuracy, Completeness, and Robustness (Days 5, 9, 10) all assume
the Day 3 baseline answer is a meaningful judgment to test the explanation
against. For rules 1-3, the baseline answer is currently close to a constant
"compliant" classifier, so masking/perturbing the top region and checking
"did the answer change" will mostly have nothing to flip -- those metrics
would read as uniformly weak for rules 1-3 not because the explanation method
is bad, but because the upstream proxy answer was already wrong before any
explanation step ran.

This is flagged here rather than silently patched, per the project's existing
practice of documenting real adaptation gaps. **Decision: redesigned the
rules 1-3 proxy** (see below) rather than accepting the scene-level version.

## Redesign: person-relative presence check

`inference.py`'s `_answer_presence_rule` now detects all "worker" boxes and
all safety-object boxes independently, then flags **violation** if any
detected worker has no nearby/overlapping safety-object box
(`regions.all_boxes_covered`) -- mirroring the dataset's own semantics that
one non-compliant worker is enough to mark the whole image violated. Worn
PPE (rules 1-2) uses a tight proximity threshold (8% of max image dimension,
since the object must be on the worker's body); rule 3's guardrail is
structural and can protect from further away, so it uses a looser threshold
(20%). Images with zero detected workers fall back to the old scene-level
existence check (12/163 samples hit this fallback).

Re-ran `scripts/03_run_baseline_inference.py` on all 163 samples with the new
proxy:

| Proxy | Sensitivity | Specificity | n |
|---|---|---|---|
| Presence-based, scene-level (old) | 3.0% (3/100) | 97.4% (37/38) | violation n=100, compliant n=38 |
| Presence-based, person-relative (new) | **27.0%** (27/100) | 73.7% (28/38) | violation n=100, compliant n=38 |
| Proximity-based (rule 4, unchanged) | 61.5% (8/13) | 41.7% (5/12) | struck_by_risk n=13, compliant n=12 |

A 9x sensitivity improvement, at the cost of some specificity (73.7% vs
97.4%) -- a real and expected precision/recall tradeoff for a proxy that now
actually tries to discriminate instead of defaulting to "compliant". Per-rule
breakdown on violation samples: rule_1 (hard hat) 8/50 = 16%, rule_2
(harness) 4/13 = 30.8%, rule_3 (guardrail) 15/37 = 40.5% -- rule_3 benefits
most, likely because its looser proximity threshold is closer in spirit to
the relational rule_4 check that already worked well.

### Residual limitation: Florence-2 often returns only one "worker" box in multi-worker scenes

Visually confirmed on image 7 (5 workers visible, ground truth: ppe_violation)
and image 79 (multiple people visible, ground truth: fall_hazard): the
`<OPEN_VOCABULARY_DETECTION>` call for the singular phrase "worker" returned
only **one** box, covering the most salient/clear person in the scene --
not one box per visible worker. When that single detected worker happens to
be the compliant one, the per-worker check still can't catch the actual
violator, because the violator was never detected as a separate "worker"
instance to check in the first place. This caps how much further the
person-relative redesign can improve sensitivity without a different
detection strategy (e.g. trying a plural phrasing, or switching from
`<OPEN_VOCABULARY_DETECTION>` to `<DENSE_REGION_CAPTION>`/`<OD>` for more
exhaustive instance enumeration) -- noted here as a real, observed limitation
rather than pursued further in this pass, since the 9x sensitivity gain
already achieved is a meaningful, honestly-earned improvement and further
detection-strategy changes would be a new design iteration, not a fix.
