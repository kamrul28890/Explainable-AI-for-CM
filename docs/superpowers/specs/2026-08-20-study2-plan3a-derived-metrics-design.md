# Study 2 Plan 3a — Derived Metrics (Stages 3, 7, 10) — Design

**Date:** 2026-08-20
**Status:** approved
**Covers:** Stage 3 (explanation regions), Stage 7 (Metric 4 — Efficiency),
Stage 10 (Metric 7 — Explanation Correctness).
**Predecessor:** Plan 2 complete. `results/baseline_qwen7b-4bit.csv` holds 3,882
verified rows, 0 missing, 100% parse.

## Goal

Compute every metric that needs no further model inference, from data already on
disk. These run on CPU, so they do not compete with the GPU battery, and they
answer the question that decides whether the remaining ~47 GPU-hours are
worth spending: **are the explanations any good?**

## Why these three, and why not Stage 9

Stages 3, 7 and 10 depend only on `baseline_*.csv` and dataset metadata.

**Stage 9 (Bounded Completeness) is deliberately excluded.** The architecture
document defines it as combining Stages 3 and 4, and its three-way verdict needs
the token-ablation flip to separate `explanation_supported` from
`explanation_weak`. Without Stage 4 it could only report `no_usable_explanation`,
which Stage 3 already gives. It belongs to Plan 3b.

## Context that shapes the design

Stage 2 measured sensitivity 91%, specificity 41%, false-alarm rate 59%. The
model answers YES to 70% of everything. That makes Metric 7 the load-bearing
metric of this plan: when a model says YES this often, being *right* is weak
evidence of having *understood*, and Metric 7 is the only measure in the battery
that can tell the difference. Its headline — the **right-answer-wrong-reason
rate** — is the deliverable.

---

## Contract

- **New:** `src/xai_vlm/metrics/regions.py`, `efficiency.py`, `correctness.py`,
  `text_similarity.py`; `scripts/03_regions.py`, `07_efficiency.py`,
  `10_explanation_correctness.py`.
- **Unchanged:** everything in Plan 1 and Plan 2. No re-running of Stage 2.
- **No GPU.** `sentence-transformers` 5.7.0 is already installed and runs on CPU.
- **No new pip dependency.** One new *model weight* (the embedding model), pinned
  by revision in `requirements/models.lock` like every other weight in this study.

---

## Stage 3 — Explanation regions

**Input:** `results/baseline_{model}.csv`.
**Output:** `results/regions_{model}.csv`.

Columns: `image_id, rule_id, verdict, source, region, image_w, image_h`,
where `source ∈ {model, none}`.

**Two jobs, not one.**

1. Classify each row's region as `model` (a box was given) or `none`.
2. **Record image dimensions.** Metric 7 must compare a normalised annotation box
   against an absolute-pixel model region, and neither can be converted without
   width and height. Stage 2 does not store them.

Dimensions come from a header-only pass: `PIL.Image.open` on the stored bytes
reads the size without decoding pixel data, so this is I/O-bound and cheap
compared with a full decode of 2,556 images.

**No patch snapping here.** Snapping a region outward to the patch grid needs the
per-image grid (21×29 landscape, 33×18 portrait), and it is only ever consumed by
the masking stages, which load the images anyway. Doing it here would force a
second full decode and create a second place for the grid to be computed
differently. Plan 3b snaps, at the point of use.

**Headline:** the **no-region rate** — the share of cases where the system offered
no explanation at all. A model with high accuracy and a high no-region rate is not
explainable, whatever its accuracy.

---

## Stage 7 — Metric 4: Efficiency

**Input:** `latency_ms` in `results/baseline_{model}.csv`.
**Output:** `results/efficiency_{model}.csv`.

Report median, p90, mean and max, overall and per rule. Median and p90 are the
headline; the mean is included only because it is conventional, and the p90 is
what determines whether a deployment feels responsive.

**Two figures are reported side by side, and this is not optional.** The measured
in-run median is 10.1 s. The *same images*, answered standalone, take 3.1 s. That
3.5× gap is real, reproducible, and unexplained — ruled out: VRAM (peak 6.97 GB of
8, stable), system RAM (18 GB free), image dimensions, and reply length. The
leading unverified hypothesis is CPU contention between parquet decoding and
Qwen's CPU-side preprocessing, which falls inside the timed call.

Quoting only the standalone figure would understate deployment cost by 3.5×;
quoting only the in-run figure attributes to the model a cost that may belong to
the harness. **In-run is primary** — it is what an actual batch run costs — with
the standalone figure and this caveat reported beside it.

Also report peak VRAM (6.97 GB of 8.0) and the 477 s maximum, which is a single
outlier and flagged as unexplained rather than smoothed away.

---

## Stage 10 — Metric 7: Explanation Correctness

**Input:** `baseline_{model}.csv`, `regions_{model}.csv`, dataset annotations.
**Output:** `results/explanation_correctness_{model}.csv`.

### Joining to ground truth

For each row where the model answered YES, fetch the dataset's
`{rule_id}_violation` record, giving `reason` (human text) and `bounding_box`
(a **list** of normalised boxes). This is a metadata-only scan, reusing the
`scan_records` pattern from Stage 1 over `config.SPLITS`.

### The three signals

**1. Semantic similarity.** Cosine similarity between embeddings of the model's
`rationale` and the human's `reason`, via `sentence-transformers` on CPU with
`all-MiniLM-L6-v2`, pinned by revision in `models.lock`.

Lexical overlap is not adequate here and the architecture document's own worked
example proves it: *"The mechanic in front of the truck is not wearing a hard
hat"* against *"The man working beside the truck cab has no helmet on"* is meant
to score 0.88 while sharing almost no vocabulary. TF-IDF would score it near zero
and mark a correct explanation incorrect.

**2. Hazard-type match.** The model was asked about one specific rule. Its
rationale is embedded against canonical one-sentence descriptions of the four
hazards, and the nearest must equal the asked rule's hazard. This catches
answering YES about PPE while describing an unguarded trench.

**3. Spatial agreement.** IoU between the model's region and the human's box.

**Coordinate conversion is load-bearing.** Annotation boxes are normalised 0–1;
model regions are absolute pixels. The architecture document flags getting this
wrong as a common, easily-missed bug that places every box in the top-left corner.
Conversion is explicit, one-way (model pixels → normalised, using `image_w` and
`image_h` from Stage 3), and directly unit-tested.

**Multiple human boxes.** `bounding_box` is a list. The score is the **maximum**
IoU over all human boxes for that rule: the model is asked to point at *the single
most relevant area*, so agreeing with any one annotated instance is success.
Averaging would penalise a correct answer on a photo with three violations.

### The verdict

```
CORRECT  iff  cosine_similarity >= 0.60
         AND  hazard_match
         AND  IoU >= 0.30
```

**Conjunctive, and pre-registered here before any value is computed.** The
spatial signal is the anchor: text similarity is fuzzy, box overlap is arithmetic,
and a rationale that reads plausibly while pointing at the wrong side of the
photograph must fail. A weighted score would let a high text score outvote a
completely wrong location, which is the exact failure geometry exists to catch.

Thresholds are taken from convention — 0.60 is a standard paraphrase floor for
MiniLM-class bi-encoders, 0.30 a deliberately permissive IoU given that the model
is asked for one region and the annotation may mark several. **They are not to be
tuned against our results.** A test asserts they are module constants, so
adjusting them is a visible, deliberate edit rather than a quiet slide.

All three raw signals are written to the CSV beside the verdict, so any reader can
re-threshold without re-running anything.

### Undefined cases are counted, never coerced

A row is **undefined** for this metric when the model answered NO (no rationale to
grade), gave no region (no spatial signal), or the annotation has no box. These
are excluded from the verdict and reported as their own rate.

Scoring them as `INCORRECT` would inflate the failure rate with cases the metric
does not cover; scoring them `CORRECT` would do the reverse. Either is a silent
lie about coverage — the failure mode this study keeps encountering.

### Headline outputs

- **Right-answer-wrong-reason rate**: verdict YES, ground truth YES, explanation
  INCORRECT. The most important failure mode in safety-critical AI, and invisible
  to every other metric here.
- Verdict distribution, and each signal's distribution.
- **All of it split by `quality_of_info`** (§5.5). If scores differ sharply
  between "rich info" and "poor info", the annotations are the limiting factor
  rather than the model.

### Stated limitation

Embedding similarity plus structured checks is weaker than expert review, and this
is a local-compute substitute for an LLM judge. The architecture document's
mitigation stands: the §5.5 human audit of 100 cases, reporting agreement between
the automatic verdict and the human one so readers can calibrate trust. That audit
is not part of this plan.

---

## Testing

All CPU, no model weights required in the unit suite.

- **`text_similarity.py` is isolated behind a small interface** so `correctness.py`
  is tested against a fake embedder with hand-chosen vectors. The verdict logic is
  then exact arithmetic rather than a floating-point guess about what MiniLM does.
- **The architecture document's two worked examples become literal test cases**:
  (0.88, PPE match, IoU 0.63) → CORRECT; (0.11, no match, IoU 0.03) → INCORRECT.
- **Coordinate conversion is tested directly**, including the top-left-corner bug:
  a normalised box `[0.5, 0.5, 0.6, 0.6]` on a 1200×900 image must not compare
  equal to a pixel box `[0.5, 0.5, 0.6, 0.6]`.
- **Max-over-boxes** is tested with a model region matching the second of three
  annotated boxes.
- **Undefined classification** is tested for each of its three causes.
- **Thresholds are asserted as constants**, with the test naming why: they are
  pre-registered and must not drift.
- One `@pytest.mark.gpu`-free integration test runs the real embedder on the §6.3
  example pair and asserts similarity > 0.60, so the pinned model is verified to
  behave as the design assumes.

## Out of scope

- **Stage 9**, Bounded Completeness — needs Stage 4 (Plan 3b).
- **Patch snapping** — done at point of use in Plan 3b.
- **The §5.5 human audit** — required to calibrate this metric, not part of it.
- **Comparison models** — the scripts take `--model`, but only `qwen7b-4bit` has
  a baseline.

## Risks

1. **The embedding model is a new pinned weight.** It must be downloaded and its
   revision recorded, or results are unreproducible. Mitigated by the same
   `models.lock` discipline as the VLM, and by the integration test.
2. **0.60 may prove poorly calibrated for this domain.** Construction-safety
   sentences are formulaic and may score higher than general paraphrase pairs,
   compressing the range. Mitigated by writing all raw signals to the CSV, so
   re-thresholding costs nothing, and by reporting the distribution rather than
   only the verdict.
3. **`quality_of_info` is 47% "poor info".** If correctness differs sharply across
   that split, the ceiling is annotation quality, not the model. This is a finding
   rather than a defect, and the split is mandatory for exactly that reason.
