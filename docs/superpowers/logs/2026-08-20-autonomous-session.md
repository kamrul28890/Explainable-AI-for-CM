# Autonomous session log — 2026-08-20 → 2026-08-21

Running log kept at the user's request while they sleep. Newest entries at the
bottom. Every claim here is either a command's recorded output or is labelled as
inference.

**Standing instructions for this session**
- Plan, implement, keep both CPU and GPU busy.
- Scientific method for methodology; search the internet when stuck.
- Everything is destined for a research paper — design accordingly.
- Log everything.

**Operating rules I am holding myself to** (from the day's failures, see
`2026-08-18` history):
1. A run exiting 0 is not evidence of success. Verify coverage against the
   manifest, every time.
2. A plausible number is not a verified number. Where a result could be an
   artefact, hand-check the artefact hypothesis before reporting.
3. Thresholds are pre-registered and never tuned against observed values.
4. Undefined is not failure — count it separately.

---

## State at session start

**Committed and verified**
| artefact | status |
|---|---|
| Stage 0 gate | GO |
| Stage 1 manifest | frozen, 3,882 pairs / 2,556 images, sha `af91ed28` |
| Stage 2 baseline | complete, 3,882 rows, 0 missing, 0 errors, 100% parse |
| Stage 3 regions | no-region 21.9% overall, 3.5% among YES |
| Stage 7 efficiency | median 10.1 s, p90 26.5 s |
| Stage 10 correctness | 2.4% correct; right-answer-wrong-reason 97.6% |

**Headline findings so far**
- Sensitivity 91%, specificity 41%, false-alarm rate 59%.
- Explanation correctness 2.4%, bound by IoU (6.8% pass) rather than text
  (similarity 56.9%, hazard match 91.2%).
- Model regions are tiny — median 0.64% of image area.

**Outstanding**
- Plan 3a Task 7 (docs).
- Plan 3b: spec written, no plan document, not implemented. ~17 h GPU.
- Plan 3c: not designed.
- Paper: no structure yet.
- Open threads: 3.5× standalone-vs-in-run latency gap; 477 s outlier;
  fingerprint-check-before-model-load.

**Plan for the session**
1. Get Stage 4 built and launched — the GPU is idle and this is the long pole.
2. CPU work alongside: docs, Stage 9, Plan 3c design, paper scaffolding.
3. Log as I go.

---

## Entries

### 01:20 — Stage 4 launched; reporting layer built

**Stage 4 (descriptive accuracy)** implemented TDD (7 driver tests + 7 flip-logic
tests, all green first run) and smoke-running on GPU. Measured latency is worse
than projected: ~34 s per row (two masking calls), against the 20 s the 17 h
estimate assumed. Projection on n=2 is **28.7 h**, not 17 h. n=2 is not a basis
for a decision; waiting for the 20-row smoke before committing.

**Reporting layer** (`metrics/reporting.py`, `scripts/11_report.py`) built so the
paper has one command and one source of truth. Reuses the pilot's `bootstrap_ci`
and `MIN_N_FLOOR=30` rather than reimplementing, so both studies report on the
same standard.

**MAJOR FINDING, surfaced only by per-rule reporting with intervals:**

```
sensitivity by rule
  rule_1 (PPE)              93.2% [91.7, 94.7]  n=1001
  rule_2 (fall protection)  97.6% [94.0, 100.0] n=84
  rule_3 (edge protection)  98.8% [97.1, 100.0] n=172
  rule_4 (plant proximity)  30.0% [20.0, 41.4]  n=70
```

The pooled 91% sensitivity **hides a rule the model is nearly blind to**. rule_4
is struck-by risk — a worker in the danger zone of operating machinery, one of
construction's "fatal four". The CI does not approach any other rule's. A model
that answers YES to 70% of everything still misses 70% of these.

This was invisible in every figure reported until now. It is a headline result and
a safety-critical one: the hazard class the model handles worst is among the most
lethal on a real site.

*Inference, not yet tested:* rule_4 is the only rule with `context_fields`
(`excavator`), so it is the only one whose negative controls are context-gated.
Whether that interacts with the low sensitivity is worth checking, but the
sensitivity figure is computed over annotated violations, so the control policy
should not affect it directly.

**Fixed:** em-dashes in report section labels mangled on the cp1252 console;
switched to ASCII hyphens in labels.

### 01:35 — Stage 4 is far slower than budgeted; subsample pre-registered

**Measured cost.** Smoke shows ~96 s per row wall-clock against ~39 s of recorded
model time. The gap is real work that `latency_ms` never counted: it times only
`model.generate()`, while `_build_inputs()` — Qwen's **slow** image processor —
runs before the clock starts, twice per row.

The warning was in every run log from the beginning:
`Using a slow image processor as use_fast is unset`.

**Deliberately NOT switching to the fast processor.** The same warning states it
"will result in minor differences in outputs". Stage 2's 3,882 baseline rows were
produced with the slow processor; changing it now would make Stage 4 incomparable
with the baseline it is measured against. Recorded as a finding for future runs
and for the paper's efficiency section, not applied mid-study.

**Consequence.** Full Stage 4 is ~81 h, against the architecture doc's ~11 h
budget (which assumed 4 s/call). That is not a schedule — it would consume the
time Stages 5, 6 and 8 need.

**Decision, made autonomously and pre-registered here before any flip rate
exists.** Subsample rather than cut. The doc says never to cut Stage 4; it does
not require running every row. Precision for a proportion near 0.5:

```
n=400   +/- 4.9 points
n=800   +/- 3.5 points
n=2621  +/- 1.9 points   (full set, ~50 extra GPU-hours)
```

Sample: **800 YES + 400 NO controls = 1,200 rows, ~32 h**, seed 42.

**Sampling design corrected once before use.** Purely proportional stratification
gave rule_4 only 21 YES rows — below the n=30 floor, and therefore unreportable.
rule_4 is the subgroup the baseline found the model nearly blind to (30%
sensitivity against 93–99%), so it is the one most worth measuring. Switched to
floor-then-proportional (`PER_RULE_FLOOR = 40`):

```
YES: rule_1 352, rule_2 119, rule_3 281, rule_4 48   (none below floor)
NO : rule_1 101, rule_2 118, rule_3  13, rule_4 168  (rule_3 pool is only 13 — unavoidable, flagged)
```

Draw is seeded and verified deterministic.

**Also built:** Stage 9 (`completeness.py` + `scripts/09_completeness.py`, 6 tests)
and `report/PAPER-OUTLINE.md` — a findings register where every claim carries its
source file and n, or is marked not-yet-measured.

### 02:10 — Root cause found: autograd graph retained in the ablation path

**The bug.** `_multimodal_embeds()` ran `self.model.visual(...)` — a full
vision-tower forward — **outside** `torch.inference_mode()`. It therefore built an
autograd graph and retained every intermediate activation for a backward pass that
never comes. The generate call beneath it was correctly guarded; the embeds
computation above it was not.

**Measured, per call, one 1200x900 image:**

```
                       before fix          after fix
answer()               3.1s / 6.97 GB      unchanged
answer_masked black    6.3s / 6.97 GB      unchanged
answer_masked tokens  54.5s / 22.65 GB     2.8s / 6.96 GB
```

~16 GB of retained activations on an 8 GB card, served from Windows shared memory
at a 19.5x speed penalty.

**My first hypothesis was wrong and the probe that disproved it found the answer.**
I predicted a missing KV cache and expected time to scale quadratically with
`max_new_tokens`. It did not (56.0 / 56.1 / 56.5 s at 32 / 64 / 128), and
`use_cache=True` changed nothing. But in that probe `input_ids` — measured at 3.1 s
in the clean run — showed 56 s, purely because I had materialised `embeds`
beforehand and the leaked graph was still resident. The contamination was the
evidence.

**Comparability verified, not assumed.** `inference_mode` disables gradient
tracking and version counting; the arithmetic is identical. Re-ran the 6 smoke
rows produced by the OLD path: **verdicts identical 6/6**. Unlike the fast image
processor (which explicitly warns it changes outputs, and which I left alone),
this is safe to apply mid-study.

**Consequence: the pre-registered subsample is retired.** It existed to survive an
81 h run. Full Stage 4 is now ~10 h, so all 3,031 rows will run — strictly better
than sampling 1,200. The `--sample` path stays in the code, documented, unused.

**Also affects Stages 5, 6 and 8**, which all use token ablation.
