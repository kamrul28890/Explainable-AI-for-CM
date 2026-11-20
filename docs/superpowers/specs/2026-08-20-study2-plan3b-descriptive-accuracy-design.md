# Study 2 Plan 3b — Descriptive Accuracy and Bounded Completeness — Design

**Date:** 2026-08-20
**Status:** approved
**Covers:** Stage 4 (Metric 1 — Descriptive Accuracy), Stage 9 (Metric 6 —
Bounded Completeness).
**Predecessor:** Plan 2 complete; Plan 3a specced. Stage 4 additionally needs
`regions_{model}.csv` from Plan 3a Stage 3.

## Goal

Answer the question the whole study is built around: **when the model points at
something, was that thing actually load-bearing?**

The architecture document calls Stage 4 the primary faithfulness measurement and
says explicitly that it must never be cut to save time.

## Why this matters more after Stage 2

Stage 2 measured sensitivity 91%, specificity 41%, false-alarm rate 59%. A model
answering YES to 70% of everything gets many verdicts right by disposition rather
than by evidence. Stage 4 is the test that disposition cannot pass: if the model's
YES does not depend on the region it named, then the region is decoration and the
explanation is not an explanation.

## Measured scope

From the completed baseline:

```
YES with region   2,621    core maskable set
NO  with region     410    artefact control
YES without region   94    -> no_usable_explanation (Stage 9)
no-region rate     21.9%
```

**6,062 model calls** (2 masking methods × 3,031 rows), ≈ **17 h** at the measured
10.1 s/call.

---

## Contract

- **New:** `src/xai_vlm/metrics/descriptive_accuracy.py`, `completeness.py`;
  `scripts/04_descriptive_accuracy.py`, `09_completeness.py`.
- **Reuses unchanged:** `runner.py` (resume, fingerprint, guards), `masking.py`
  (patch alignment), `backends/qwen.py` `answer_masked(...)`, which already
  implements both `black` and `tokens`.
- **Depends on:** `regions_{model}.csv` (Plan 3a Stage 3) for region source and
  image dimensions.
- **Does not re-run Stage 2.**

---

## Stage 4 — Descriptive Accuracy

### Which rows

Every row with a region, in both verdict classes:

- **2,621 YES + region** — the measurement.
- **410 NO + region** — the control, and it is not padding.

If masking only removes information, a NO answer cannot rationally become a YES.
The **NO→YES rate is therefore an artefact floor**: it measures how much the
intervention perturbs the model by itself, independent of any evidence being
removed. Without it, a high YES→NO flip rate cannot be distinguished from a model
that simply reacts to any edit of its input.

### Procedure per row

1. Load the image; take the region from `regions_{model}.csv`.
2. **Snap the region outward to the patch grid** via `masking.py`, using the
   per-image grid from `backend.patch_grid(image)`. A mask cutting a patch in half
   leaves partial, ambiguous signal in it.
3. `answer_masked(image, rule_id, region, method="black")` → record verdict.
4. `answer_masked(image, rule_id, region, method="tokens")` → record verdict.
5. Record a flip flag per method against the baseline verdict.

**Patch snapping happens here, not in Stage 3.** This is the point of use, images
are already loaded, and computing the grid in one place removes any chance of two
stages disagreeing about it.

### Row keying and resume

`runner.run_stage` keys work on `(image_id, rule_id)`. Rather than widen that
contract to include the masking method, **one `work_fn` call performs both
maskings and emits one row carrying both flips.**

Resume granularity is therefore the pair — two model calls, at most ~20 s of
rework after a crash. The alternative, a three-part key, would change an interface
that Stages 5, 6 and 8 will also consume, to save twenty seconds.

### Fingerprint

The Stage 2 fingerprint fields, plus:

| field | why |
|---|---|
| `baseline_sha256` | regions derive from the baseline; re-running Stage 2 must invalidate Stage 4 |
| `regions_sha256` | the direct input |
| `masking_methods` | `["black", "tokens"]` — changing the set changes the measurement |

### Output

`results/descriptive_accuracy_{model}.csv`, one row per masked pair:

```
image_id, rule_id, baseline_verdict, region, region_snapped,
verdict_black,  flip_black,
verdict_tokens, flip_tokens,
methods_disagree, latency_ms_black, latency_ms_tokens, error
```

### Interpretation, fixed in advance

- **`flip_tokens` is the primary figure.** Token ablation deletes the image tokens
  for those patches before the language model runs, so no out-of-distribution
  pixels exist. Nothing in the image is edited.
- **`flip_black` is reported beside it** for comparability with prior work, and
  **the gap between them is itself a result** — it quantifies how much the
  black-rectangle artefact inflates the apparent flip rate.
- **`NO→YES` rate on the control rows is the artefact floor.** A YES→NO flip rate
  that fails to clear it is not evidence of faithfulness.
- **`methods_disagree` rows are flagged, never averaged.** Disagreement between an
  artefact-free and an artefact-prone intervention is the doc's non-monotonicity
  diagnostic and diagnoses residual artefact.
- All results split by `quality_of_info` (§5.5).

---

## Stage 9 — Bounded Completeness

No new model calls. Joins `regions_{model}.csv` and
`descriptive_accuracy_{model}.csv`:

| verdict | condition |
|---|---|
| `no_usable_explanation` | no region was returned |
| `explanation_supported` | region given, **token ablation** flipped the answer |
| `explanation_weak` | region given, removing it changed nothing |

**Output:** `results/completeness_{model}.csv`.

**Headline:** the `explanation_supported` share, computed from the token-ablation
flip specifically, to avoid crediting the black-rectangle artefact.

`explanation_weak` is the interesting category and should be reported as
prominently as `supported`: the model named a region, that region turned out not
to matter, and a reader shown only the region would have been misled about why the
system decided what it did.

---

## Testing

**CPU, no GPU, against `FakeBackend`:**

- Flip detection: a `FakeBackend` scripted YES → NO must record `flip=True`; YES →
  YES must record `flip=False`.
- Both methods run per row, and both are recorded — asserted via `FakeBackend.calls`.
- Region snapping is applied before masking, asserted on the recorded
  `region_snapped` against a known grid.
- `methods_disagree` is set exactly when the two flips differ.
- Fingerprint includes `baseline_sha256`; changing the baseline aborts resume.
- Crash-and-resume over a fake manifest yields exactly one row per pair, no
  duplicates — the same guarantee proved for Stage 2.
- Stage 9's three-way classification is a pure function, tested exhaustively over
  its input combinations, including the NO-verdict and no-region cases.

**GPU validation** before the full run: `--limit 20`, confirming both masking
methods execute, peak VRAM stays within budget, and snapped regions land on patch
boundaries.

## Out of scope

- **Inpainting and counterfactual insertion.** The architecture document lists
  both as subset extras. Neither is needed for the headline faithfulness figure,
  and both can be added later without redoing any of this.
- **Larger-mask non-monotonicity sweeps.** The `methods_disagree` flag covers the
  diagnostic need at no extra cost.
- **Comparison models.** Structure supports `--model`; only `qwen7b-4bit` has a
  baseline today.

## Risks

1. **17 h is long enough that something will interrupt it.** Mitigated by the
   Plan 2 machinery: append-only checkpointing, `(image_id, rule_id)` resume,
   fingerprint verification, and the coverage check that makes a partial run exit
   non-zero rather than report success.
2. **Token ablation is the primary figure and the less-tested path.** It passed 12
   GPU tests in Plan 1, but at far smaller scale. The `--limit 20` smoke run must
   confirm both methods before committing 17 h.
3. **A high artefact floor would undercut the headline.** If NO→YES flips are
   common, the YES→NO rate means less than it appears. This is a finding rather
   than a defect, and it is precisely why the control rows are included.
4. **Peak VRAM was 6.97 GB of 8.0 in Stage 2.** Token ablation replicates the
   `masked_scatter` merge and may allocate more. The smoke run must measure it.
