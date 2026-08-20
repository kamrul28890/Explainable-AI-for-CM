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
