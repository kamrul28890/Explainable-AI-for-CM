# Study 2 — paper outline and findings register

**Status:** living document. Every claim carries its source file and sample size,
or is explicitly marked as not yet measured.

**Target:** a venue concerned with trustworthy/explainable AI in safety-critical
deployment. The contribution is not "a VLM scores X" but **what it takes to
measure explanation quality honestly, and what that measurement reveals that
accuracy hides.**

---

## The argument in one paragraph

A quantised vision-language model achieves 91% sensitivity on construction-site
hazard detection — a figure that would pass most deployment reviews. Measuring the
same model against seven explanation-quality metrics shows that number to be
close to meaningless: it answers YES to 70% of everything, raises a false alarm on
59% of hazard-free checks, is nearly blind to one of the deadliest hazard classes,
and when it is right it can almost never point at why. The paper's contribution is
the measurement protocol that makes those failures visible, and the demonstration
that each of them is invisible to conventional accuracy reporting.

---

## Findings register

### F1 — Accuracy is bought with false alarms
`results/baseline_qwen7b-4bit.csv`, n=3,882

```
sensitivity        90.9% [89.3, 92.4]  n=1,327
specificity        40.9% [39.0, 42.9]  n=2,555
false-alarm rate   59.1% [57.1, 61.0]  n=2,555
YES share          69.9% [68.5, 71.4]  n=3,882
```

Balanced accuracy 66% against 50% for an always-YES model. **Six false alerts per
ten genuine ones** — an alarm-fatigue machine in deployment.

*Why it matters for the paper:* the sensitivity figure alone would be reported as
a success. The negative-control design is what exposes it, and negative controls
are absent from most VLM safety evaluations.

### F2 — The pooled figure hides a near-blind hazard class
Same file, per-rule.

```
rule_1  PPE                93.2% [91.7, 94.7]   n=1,001
rule_2  fall protection    97.6% [94.0, 100.0]  n=84
rule_3  edge protection    98.8% [97.1, 100.0]  n=172
rule_4  plant proximity    30.0% [20.0, 41.4]   n=70
```

`rule_4` is **struck-by risk** — a worker inside the danger zone of operating
machinery — one of construction's "fatal four" causes of death. Its interval does
not approach any other rule's. A model that says YES to 70% of everything still
misses 70% of these.

*Why it matters:* this is the paper's sharpest single result. It is invisible in
the pooled 91%, and it is invisible without per-class intervals. The hazard the
model handles worst is among the most lethal.

*Open:* is this a detection failure or a question-wording failure? `rule_4` is the
only rule with a context gate (`excavator`). Testing this needs a prompt ablation,
not yet run.

### F3 — The model describes hazards well and locates them badly
`results/explanation_correctness_qwen7b-4bit.csv`, scored n=1,191

```
explanation correct         2.4% [1.6, 3.4]
  similarity >= 0.60       56.9% [54.1, 59.8]
  hazard match             91.2% [89.4, 92.8]
  IoU >= 0.30               6.8% [5.4, 8.3]     <- binding constraint
right answer, wrong reason 97.6% [96.6, 98.4]
```

*Verification, which belongs in the paper:* a 6.8% IoU rate is also the signature
of a normalised-vs-pixel coordinate error. It was hand-checked against raw boxes
before being reported — high-IoU rows match the annotation to three decimals
(`0.706, 0.470, 0.824, 0.865` vs `0.70, 0.47, 0.82, 0.90`).

*Three caveats that must travel with it:*
1. Model regions are tiny (median 0.64% of image area); IoU is severe for small
   boxes.
2. 1,430 rows are unscoreable — **a false alarm has no correct explanation**, so
   this metric structurally cannot penalise the model's largest failure mode.
3. Zero IoU may mean the model found a real violation nobody annotated; 47% of the
   corpus is self-flagged "poor info".

### F4 — Faithfulness (Stage 4) — IN PROGRESS
Does removing the named region change the answer? Includes an **artefact floor**:
the NO→YES rate, which cannot rationally be non-zero if masking only removes
information, and therefore measures how much the intervention perturbs the model
by itself.

### F5 — Cost
```
median 10.1 s   p90 26.5 s   max 477 s   n=3,882
```
Per-rule medians 10.0–10.3 s — near-identical across four questions of differing
complexity, indicating a roughly constant per-call overhead.

**Unresolved and reported as such:** the same images answered standalone take
3.1 s. Ruled out: VRAM, system RAM, image size, reply length.

---

## Methodological contribution

This is the part most likely to survive review, and it is grounded in failures
that actually occurred during this study, not hypotheticals.

**M1 — Silent incompleteness.** A run exited 0, reported 0 errors, 100% parse, and
a wholly plausible summary (sensitivity 89%, specificity 43%) while having covered
**1,231 of 3,882 rows**. Parse-rate and latency guards cannot catch this: they only
observe rows that were *attempted*. Coverage must be verified against the manifest
as a separate artefact-level check.

**M2 — Thresholds calibrated on unrepresentative samples.** A latency abort was
derived from a 39-image gate set and sat 10% above the real workload's baseline,
so it aborted healthy runs. Calibrate on the workload, not the pilot.

**M3 — Configuration drift between stages.** A split list defined in two places
drifted; one stage sampled both dataset splits while the next read only one.
Single source of truth, or the two cannot disagree loudly.

**M4 — Plausible numbers need adversarial checks.** Two results in this study
(436 positives; 6.8% IoU) were both correct-looking and both indistinguishable
from a specific bug. Each was resolved by comparing against a pre-registered
external artefact, not by internal consistency.

**M5 — Undefined is not failure.** 2,691 of 3,882 rows are undefined for metric 7
for four distinct reasons. Collapsing them into "incorrect" would have produced a
much worse-looking and completely wrong result.

**M6 — Correct-but-slow is a failure mode that no correctness test catches.**
The token-ablation path — the study's primary artefact-free intervention — ran the
vision tower outside `torch.inference_mode()`, retaining ~16 GB of autograd
activations for a backward pass that never comes. On an 8 GB card the driver
served them from shared system memory rather than raising OOM.

```
answer_masked(tokens)   54.5 s / 22.65 GB   ->   2.8 s / 6.96 GB
```

Every answer it produced was **correct** — verified verdict-identical on rows
computed before and after the fix. No test failed, no number was wrong, nothing
was silently dropped. The only symptom was a 19.5x cost, and the only reason it
was caught is that peak VRAM is instrumented and a 22.65 GB reading on an 8 GB
card was implausible enough to chase.

*Two transferable points.* First, **instrument resources, not just outputs**: a
harness that records only accuracy and latency cannot distinguish "this model is
expensive" from "our harness is broken". The 3.5x standalone-vs-in-run gap (F5)
went unexplained for a day precisely because it was visible only as time.

Second, **the diagnostic detour paid for itself**: an hour of measurement replaced
a 32-hour workaround (a pre-registered subsample, drafted and then retired) with a
one-line fix, and the saving repeats across Stages 5, 6 and 8, which share the
intervention.

*Counter-example worth reporting beside it.* The same run log warned
`Using a slow image processor as use_fast is unset`, which is also a real speed
cost — and that one was **deliberately left in place**, because the warning states
it changes outputs and the baseline was already computed with it. Not every
available speedup is available mid-study. The distinction is whether the change
can alter a number, and that has to be checked rather than assumed:
`inference_mode` cannot, the processor can.

---

## Structure

| § | Content | State |
|---|---|---|
| 1 | Introduction — accuracy is not trust | outline |
| 2 | Related work — VLM safety eval, XAI metrics | **not started** |
| 3 | Dataset, two-stratum sampling, prevalence reweighting | data ready |
| 4 | Protocol — seven metrics, pre-registration, the harness | mostly written in specs |
| 5 | Results — F1–F5 | F1–F3, F5 ready; F4 running |
| 6 | Discussion — what accuracy hides | outline |
| 7 | Threats to validity — annotation quality, IoU severity, single model | partly written |
| 8 | Methodological lessons — M1–M5 | outline |

## Not yet measured

- Stages 5 (sparsity), 6 (stability), 8 (robustness) — Plan 3c, undesigned.
- Comparison models (`qwen3b-8bit`, `qwen3b-4bit`) — the size-comparison arm.
- The §5.5 human audit of 100 cases, without which F3's caveat 3 cannot be closed.
- A prompt ablation for F2.
