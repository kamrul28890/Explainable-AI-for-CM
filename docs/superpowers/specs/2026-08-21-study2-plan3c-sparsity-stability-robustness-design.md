# Study 2 Plan 3c — Sparsity, Stability, Robustness — Design

**Date:** 2026-08-21
**Status:** drafted autonomously during the overnight session; **not yet reviewed
by the user**. Implementation may proceed, but the pre-registered choices marked
DECISION below should be confirmed.
**Covers:** Stage 5 (Metric 2 — Sparsity), Stage 6 (Metric 3 — Stability),
Stage 8 (Metric 5 — Robustness).
**Predecessor:** Plans 3a and 3b. Stage 5 needs nothing but the baseline; Stages 6
and 8 need the manifest and the baseline.

## Goal

Complete the seven-metric battery. These three ask whether the explanation is
*focused* (5), *reproducible* (6), and *durable under real-world image
degradation* (8).

## Budget, recomputed after the inference_mode fix

The 19.5× speedup on token ablation changes what is affordable. Costs below use
the measured post-fix figures (plain answer ≈ 3 s, masked ≈ 3–6 s):

| stage | calls | estimate |
|---|---|---|
| 5 — sparsity | 3,882 × (generate + 1 forward) | ~6.5 h |
| 6 — stability | 800 × 5 | ~3.5 h |
| 8 — robustness | 1,600 × 5 | ~7 h |
| | | **~17 h total** |

Before the fix this would have been prohibitive. It is now a long weekend run.

---

## Stage 5 — Metric 2: Sparsity

**Input:** the baseline manifest rows. **Output:** `sparsity_{model}.csv`.

**Procedure.** For each row, call `backend.attention(image, rule_id)`, which
generates on the sdpa path and then reads attention from a **single teacher-forced
forward pass** over the produced sequence. Normalise the grid to 0–1 and compute:

- **Gini coefficient** — primary. 0 = attention spread evenly, 1 = all mass on one
  patch.
- **Mass inside the stated region** — the share of attention falling within the
  box the model itself named.

**Why Gini is primary.** It is size-independent. A "fraction of attention in the
top-5 patches" measure punishes large objects: a guardrail spanning the frame
*should* attract spread-out attention, and would score as unfocused. Gini
describes the distribution's shape without reference to object size.

**Mass-inside-region is a second, independent honesty check.** If the model names
a region but its attention lies elsewhere, that is direct evidence of
unfaithfulness — and it is obtained without any masking, so it does not share the
artefact concerns of Stage 4. Given Stage 10 found the model's regions agree with
annotations only 6.8% of the time, this cross-check matters: it separates "the
model looked at the right place and described it badly" from "the model never
looked there at all".

**`attention()` may return None** and that is legitimate — not every backend
exposes attention. Rows where it does are recorded as `attention_unavailable` and
excluded, with the count reported. Sparsity is reported only over rows where
extraction genuinely worked.

**DECISION — run over all 3,882 rows.** Attention costs one generate plus one
forward and needs no subset at current speed.

---

## Stage 6 — Metric 3: Stability

**Input:** a subset of the manifest. **Output:** `stability_{model}.csv`.

**Procedure.** Ask the same question **5 times with sampling decoding**, then
measure verdict agreement, region-presence agreement, and centroid drift between
runs.

**Sampling decoding is mandatory and is a deviation from the frozen `GENERATION`
config.** With `do_sample=False` the model is deterministic: five reruns would be
byte-identical and score 100% stable by construction, proving nothing. This is the
one stage that must override the study-wide decoding settings, and the override is
therefore part of the measurement rather than a drift.

**DECISION — `temperature=0.7`, `top_p=0.9`, `do_sample=True`, seeds 0–4.**
Recorded in the fingerprint as `stability_generation` so a later run cannot
silently use different sampling and be concatenated with this one.

**DECISION — subset of 800 rows**, stratified by rule with the same
floor-then-proportional scheme as Stage 4 (`PER_RULE_FLOOR = 40`), seed 42.
Stability is a property of the model, not of a particular image, so 800 rows at
five draws each is ample; the full set would cost 4× more for a narrower interval
on a quantity that is not close to any decision boundary.

**Report agreement, and report flickering separately.** A region present in some
runs and absent in others is worse than a consistently wrong region, because it is
unpredictable. It must be reported rather than averaged away or dropped as
missing.

---

## Stage 8 — Metric 5: Robustness

**Input:** a subset of the manifest. **Output:** `robustness_{model}.csv`.

**Procedure.** Apply five degradations, re-ask, and record whether the verdict
survives and how far the region moves.

| perturbation | what it tests |
|---|---|
| heavy blur | camera shake, distance, poor optics |
| low light | dusk, indoor, shadowed site |
| central occlusion | obstruction between camera and hazard |
| reduced contrast | haze, glare, cheap sensor |
| **full desaturation** | reliance on colour as a shortcut |

**Report per perturbation, never pooled.** "80% robust" hides a model that is
perfect under blur and useless in low light — and low light is precisely when real
site accidents happen.

**Desaturation deserves its own discussion in the paper.** High-visibility
clothing is defined largely by colour. A large drop under colour removal means the
model keys on hue rather than recognising the garment, which is a specific,
actionable safety finding rather than a generic robustness number. Given rule_1
(PPE) is 93% sensitive and carries most of the corpus, this is the single most
informative perturbation available.

**Survival without stability is coincidence.** An answer that survives while its
region jumps across the image is not robust. Region drift is recorded and reported
alongside survival, and a "survived" verdict with large drift is flagged.

**DECISION — subset of 1,600 rows**, same stratification scheme, seed 42.
Five perturbations each.

---

## Shared design

**All three reuse `runner.run_stage`** — resume, fingerprinting, guards, coverage
verification. Nothing new is needed there.

**Fingerprints** extend the Stage 2 set with the stage-specific fields
(`stability_generation`, `perturbations`, `subset_seed`), so a run cannot be
resumed under different settings.

**All results split by `quality_of_info`**, per §5.5.

**Every reported rate carries a bootstrap CI** via `metrics/reporting.py`, and
`scripts/11_report.py` is extended to collect these stages so the paper keeps one
source of truth.

## Testing

CPU, against `FakeBackend`, in the pattern established by Stages 2 and 4:

- Gini on hand-computed grids: uniform → 0, single-hot → near 1, and a known
  intermediate case verified by hand.
- Mass-inside-region on a grid where the answer is arithmetic.
- `attention() is None` → row recorded as unavailable, never as zero sparsity.
- Stability agreement over scripted replies with known disagreement.
- Flickering regions detected where a region appears in 3 of 5 runs.
- Each perturbation applied exactly once per row, asserted via `FakeBackend.calls`.
- Crash-and-resume yields exactly one row per key, as proved for Stage 2.

## Risks

1. **Attention extraction may be unavailable or wrong under 4-bit quantisation.**
   Stage 0 confirmed it returns a grid, but not that the grid is meaningful.
   Mitigated by reporting mass-inside-region alongside Gini: if attention were
   noise, mass-inside-region would sit near the region's area fraction, which is a
   checkable null.
2. **Sampling decoding may change the parse rate.** Non-greedy output is more
   variable in format. The existing parse-rate guard covers this and will abort if
   it falls below 95%.
3. **Perturbation strength is a free parameter.** Too mild and everything
   survives; too harsh and nothing does. Strengths must be fixed before the run
   and reported, not tuned to produce an interesting spread.

## Out of scope

- Counterfactual insertion and inpainting (Stage 4 extras).
- Comparison models — structure supports `--model`, only the primary has a baseline.
