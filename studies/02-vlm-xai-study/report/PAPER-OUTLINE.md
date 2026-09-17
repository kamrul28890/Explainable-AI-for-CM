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

### F4 — The faithfulness probe is drowned by its own artefact
`results/descriptive_accuracy_qwen7b-4bit.csv`, full working set n=3,031
(2,621 YES + 410 negative controls; 0 errors, coverage verified against the
manifest)

Does removing the named region change the answer? The design carries an
**artefact floor**: the NO→YES rate. Masking only ever removes information, so a
control that said NO cannot rationally become YES. Whatever that rate is, it
measures how much the intervention perturbs the model by itself.

```
                        signal (YES)          floor (NO→YES)
black mask        6.4% [5.5, 7.4]        27.8% [23.4, 32.2]
token ablation    5.9% [5.0, 6.8]        51.7% [46.8, 56.3]
                       n=2,621                 n=410

methods disagree 12.5% [11.4, 13.7]  n=3,031
```

**The floor is four to nine times the signal, and the intervals are disjoint.**
Masking the region the model itself named moves its answer *less* than masking
moves rows where there was nothing to find. Stage 4 therefore returns no evidence
that the stated region is load-bearing — and, more importantly, it cannot: an
intervention whose null exceeds its effect has no headroom to detect one.

Per-rule, YES rows only (black / tokens):

```
rule_1  PPE               4.8% [3.6, 6.0]    4.6% [3.5, 5.8]    n=1,239
rule_2  fall protection  14.9% [11.4, 18.7] 15.5% [11.7, 19.2]  n=343
rule_3  housekeeping      4.4% [3.2, 5.8]    2.6% [1.7, 3.6]    n=969
rule_4  excavator        22.9% [12.9, 32.9] 28.6% [18.6, 40.0]  n=70
```

The ordering is suggestive — the two rules the model handles worst (F2) are the
two whose answers move most under masking — but every one of these sits below the
27.8% floor, so none of it can be read as faithfulness.

*Why it matters for the paper:* this is a null result that only exists because
the negative control was run. Without it, "6.4% of answers flip when we mask the
explained region" is a publishable-looking faithfulness number, and it is the
same shape as numbers reported in the literature. The control turns a modest
positive finding into a demonstration that the method does not work on this
model. Reporting the floor beside the effect should be a requirement, not a
courtesy.

*The token path is the worse of the two, which inverts the design assumption.*
Token ablation was adopted as the study's *artefact-free* intervention precisely
because it leaves the image untouched (see M6). It has the highest floor of the
two: deleting the hazard word flips a NO to a YES in **51.7%** of controls. It
does not remove evidence, it replaces a specific question with a vaguer one, and
the model answers the vaguer question YES. Leaving the pixels alone did not make
the intervention neutral.

*Caveats:*
1. The floor is measured on rows where the *model* said NO, not where the
   annotation says NO. Given a 59% false-alarm rate (F1), these 410 rows are a
   selected population and may be atypically fragile.
2. A flip is a binary on a single greedy decode. The pilot's graded-score work
   (Phase 2.1) exists precisely because a boolean discards magnitude; Study 2 has
   no graded equivalent yet.
3. Stage 5's mass-inside-region is the one faithfulness probe in the battery that
   requires no intervention at all, and therefore inherits none of this. After
   F4 it carries more weight than originally planned.

### F6 — The model does look where it says it looks
`results/sparsity_qwen7b-4bit.csv`, n=3,882 (coverage verified against the
manifest: 0 missing, 0 duplicate, 0 errors; attention extracted on 3,882/3,882)

**The study's first positive faithfulness result, and the only one F4's artefact
collapse cannot touch** — this stage performs no intervention at all. Nothing is
masked and nothing is deleted; attention is read from a teacher-forced pass over
the reply the model already gave.

```
Gini                        median 0.551   p25 0.503   p75 0.602   n=3,882
mass inside stated region   median 0.0237
area fraction of region     median 0.0105   <- the chance baseline
                                     ratio  2.3x

attention exceeds area null  85.7% [84.5, 87.0]   n=3,031
```

Raw attention mass means nothing alone: a region covering half the frame should
attract half the attention by chance. The reportable quantity is each row's mass
against **its own** area fraction. On 86% of rows attention concentrates inside
the named region beyond what uniform attention would put there.

Per rule:

```
rule_1  PPE               82.3% [80.2, 84.3]  n=1,342   2.2x
rule_2  fall protection   88.6% [85.6, 91.4]  n=464     2.3x
rule_3  edge protection   88.1% [86.0, 90.1]  n=982     1.5x
rule_4  struck-by         90.1% [86.0, 93.8]  n=243     2.4x
```

**rule_4 is the sharpest finding in this table.** It is the class the model is
nearly blind to — 30% sensitivity (F2) — and it has the *highest* rate of
attention landing on target. The model looks in the right place and then answers
NO. That dissociates *looking* from *deciding*: rule_4's failure is not a
perception failure, and a fix aimed at the vision encoder would address the wrong
half of the model.

rule_3's ratio is lowest (1.5x) because its regions are by far the largest
(median area 8.75% of the frame against rule_1's 0.46%). A larger region is
easier to hit by chance, which compresses the ratio while leaving the beat-null
rate high. This is exactly why Gini is the size-independent primary and why the
null is computed per row rather than pooled.

Attention is also slightly more concentrated when the model answers YES
(Gini 0.570, n=2,715) than NO (0.516, n=1,167) — consistent with a located
hazard focusing attention, though the two distributions overlap heavily.

*Why it matters for the paper:* F6 stands in direct tension with F3. The model's
attention is on target 86% of the time, while the boxes it *names* match human
annotations only 6.8% of the time. Both can be true, and together they say the
model is internally consistent but disagrees with the annotators about where the
hazard is. That reframes F3's third caveat from a hedge into a live hypothesis:
with 47% of the corpus self-flagged "poor info", the §5.5 human audit is now the
highest-value unblocking work in the study.

*Caveats:*
1. Attention is not explanation. High mass inside the region shows the model
   attended there while producing the reply; it does not prove that attention
   caused the verdict. The causal claim is what Stage 4 was for, and Stage 4
   failed.
2. 851 of 3,882 rows have no model region (Stage 3's no-region rate) and are
   excluded from mass-inside-region. Gini is reported over all 3,882.
3. Read from the manual attention implementation, since sdpa does not support
   `output_attentions=True`. Generation still runs on sdpa; attention comes from
   a separate teacher-forced pass, so the decoded text is unaffected.

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

**M7 — An intervention's null must be measured, not argued for.** Stage 4's two
maskings were chosen on reasoning about what each one disturbs: blacking pixels
is visibly out of distribution, so token ablation was adopted as the
*artefact-free* alternative because it leaves the image untouched. Measured
against negative controls, the artefact-free option was the worse of the two by a
factor of two (51.7% vs 27.8% NO→YES), and both floors exceeded the effect they
were meant to isolate. The argument for token ablation was sound and the
conclusion was wrong, because "leaves the image untouched" is not the same as
"leaves the model's task untouched" — deleting the hazard word changes the
question being asked. No amount of reasoning about the intervention substitutes
for running it on rows where the answer is known in advance.

**M8 — A guard wired to the wrong column reports health, not failure.** The
runner's two guards were written against Stage 2 and read its column names:
latency from `latency_ms`, success from a non-empty `verdict`. Every later stage
writes different columns, and because a result row is `{**input_row, **result}`,
a stage reading its input from an earlier stage's CSV *inherits that CSV's
`verdict` column*. The consequence, measured across the battery:

| stage | latency guard | parse guard |
|---|---|---|
| 2 — baseline | works | works |
| 4 — descriptive accuracy | **dead** (writes `latency_ms_black`/`_tokens`) | **vacuous** (reads Stage 3's verdict) |
| 5 — sparsity | **dead** (no latency column) | **vacuous** (reads Stage 3's verdict) |
| 6 — stability | **dead** (no latency column) | **vacuous** (reads Stage 2's verdict) |

Three of the four stages that use the runner were unguarded, including the 70 h
Stage 4 run. Neither failure is visible as an error: a missing latency column
reads as 0.0 s, so the progress line prints a healthy `median 0.0s` and the
ceiling can never be crossed, and an inherited verdict column is always
non-empty, so the parse rate is 100% by construction.

This is M1 in a second form. M1 was a guard that could not see rows that were
never attempted; M8 is a guard that could not see the stage it was watching.
Both reported success. The fix makes each stage declare how its own health is
read (`ok_fn`, `latency_fn`) rather than inheriting Stage 2's assumptions, and
the defaults are unchanged so Stage 2 is untouched. Guards run in-process and
never touch what is written, so no recorded number changes — only whether a bad
run gets caught.

*Caveat that must travel with the Stage 4 result:* F4's run was unguarded. Its
numbers survive on the strength of coverage verification (3,031/3,031 rows,
0 errors) rather than on the latency and parse guards, which were inert
throughout.

---

## Structure

| § | Content | State |
|---|---|---|
| 1 | Introduction — accuracy is not trust | outline |
| 2 | Related work — VLM safety eval, XAI metrics | **not started** |
| 3 | Dataset, two-stratum sampling, prevalence reweighting | data ready |
| 4 | Protocol — seven metrics, pre-registration, the harness | mostly written in specs |
| 5 | Results — F1–F6 | F1–F6 ready |
| 6 | Discussion — what accuracy hides | outline |
| 7 | Threats to validity — annotation quality, IoU severity, single model | partly written |
| 8 | Methodological lessons — M1–M8 | outline |

## Not yet measured

- Stage 6 (stability) — designed in the Plan 3c spec, implemented and
  unit-tested, **not yet run on GPU**. (Stage 5 is complete: see F6.)
- Stage 8 (robustness) — designed in the Plan 3c spec, **not implemented**. Its
  design should be revisited in light of F4: if a verdict barely moves when the
  region is removed entirely, "survives blur" may be measuring the same
  insensitivity rather than robustness.
- The Plan 3c spec's pre-registered DECISIONs are **not yet reviewed**.
- Comparison models (`qwen3b-8bit`, `qwen3b-4bit`) — the size-comparison arm.
- The §5.5 human audit of 100 cases, without which F3's caveat 3 cannot be closed.
- A prompt ablation for F2.
