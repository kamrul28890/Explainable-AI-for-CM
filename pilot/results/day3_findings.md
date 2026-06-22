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
practice of documenting real adaptation gaps. **This needs a decision before
Day 4 proceeds**: either (a) accept rules 1-3 as a documented limitation and
report the rule_4-style relational proxy as the one that actually worked,
matching the explicit purpose of this pilot (to find out what does and
doesn't transfer), or (b) redesign the rules 1-3 proxy to be region/person-
relative (e.g., detect all people, detect all hard hats, and flag any person
box that does not overlap a hard-hat box) before continuing -- a real code
change, not a parameter tweak.
