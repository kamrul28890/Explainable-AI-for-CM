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
