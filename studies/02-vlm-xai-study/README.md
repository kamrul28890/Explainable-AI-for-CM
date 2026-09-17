# Study 2 — VLM Explanation Quality

Measuring whether a vision-language model's *explanation* of a construction-site
safety judgement can be trusted — not just whether its answer is right.

**Design:** [../../docs/architecture/vlm-xai-study-architecture.md](../../docs/architecture/vlm-xai-study-architecture.md)
**Running it:** [../../docs/setup/REPRODUCE.md](../../docs/setup/REPRODUCE.md)

---

## Status

| component | state |
|---|---|
| `rules.py` — the four rules as questions | ✅ implemented and tested |
| `parsing.py` — reply parsing and parse-rate tracking | ✅ implemented and tested |
| `sampling.py` — two-stratum sampling, prevalence reweighting | ✅ implemented and tested |
| `backends/base.py` — the `Backend` protocol | ✅ interface defined |
| `backends/fake.py` — scripted backend for GPU-free testing | ✅ implemented and tested |
| `backends/qwen.py` — Qwen2.5-VL implementation | ✅ implemented, 12 GPU tests pass |
| `masking.py` — patch-grid alignment | ✅ implemented and tested |
| `runner.py` — resumable stage execution | ✅ implemented and tested |
| `scripts/00_feasibility_gate.py` — Stage 0 | ✅ run; **VERDICT: GO** |
| `scripts/01_build_sample.py` — Stage 1 | ✅ run; manifest frozen |
| `scripts/02_baseline_inference.py` — Stage 2 | ✅ implemented; smoke-run on GPU |
| `metrics/geometry.py` — box conversion and IoU | ✅ implemented and tested |
| `metrics/regions.py` — Stage 3 | ✅ run |
| `metrics/efficiency.py` — Stage 7 | ✅ run |
| `metrics/text_similarity.py` — pinned embeddings | ✅ implemented and tested |
| `metrics/correctness.py` — Stage 10 | ✅ run |
| `metrics/descriptive_accuracy.py` — Stage 4 | ✅ run, full set n=3,031 |
| `metrics/sparsity.py` — Stage 5 | ✅ run, full set n=3,882 |
| `metrics/stability.py` — Stage 6 | ✅ implemented and tested; ⬜ not run |
| `metrics/completeness.py` — Stage 9 | ✅ run; ⚠️ not reportable (see F7) |
| `metrics/` — Stage 8 (robustness) | ⬜ not started (Plan 3c) |

246 tests: 234 without a GPU, 12 requiring `--gpu`.

## Results so far

Every figure below is from a committed CSV in `results/`, not a summary written
by hand.

### Stage 2 — baseline (n = 3,882)

```
sensitivity        91%   (n=1,327)
specificity        41%   (n=2,555)
false-alarm rate   59%   (1,509/2,555)
YES share          70%
parse rate        100%   (3,807 strict, 75 lenient)
```

Balanced accuracy is 66% against 50% for an always-YES model. The model is above
chance, but its 91% sensitivity is substantially bought with false alarms rather
than detection — roughly six false alerts per ten genuine ones. **Report the
false-alarm rate beside any accuracy figure**, per §5.3.

### Stage 3 — explanation regions

```
no-region rate, all rows   21.9%
no-region rate, YES only    3.5%
```

These must be reported separately. The pooled 21.9% is dominated by NO answers,
which need no region; when this model claims a hazard it almost always points at
something.

### Stage 7 — efficiency

```
median 10.1 s   p90 26.5 s   mean 13.7 s   max 477 s   n=3,882
per-rule medians: 10.0-10.3 s across all four rules
```

Per-rule medians are near-identical despite different question complexity, which
suggests a roughly constant per-call overhead rather than content-dependent cost.

**The standalone/in-run gap is unexplained and both figures must be reported.**
The same images answered alone take 3.1 s. Ruled out: VRAM (peak 6.97 GB of 8,
stable), system RAM (18 GB free), image dimensions, reply length. Leading
unverified hypothesis: CPU contention between parquet decoding and Qwen's
CPU-side preprocessing, which falls inside the timed call.

### Stage 10 — explanation correctness

```
scored 1,191 of 3,882        undefined 2,691
explanation CORRECT          2.4%
right answer, wrong reason  97.6%   (1,162/1,191)

similarity >= 0.60   56.9%
hazard match         91.2%
IoU >= 0.30           6.8%    <- the binding constraint
```

The model **describes** hazards well and **locates** them badly. 91% name the
right hazard category; two thirds have no spatial overlap with the annotation at
all.

**The geometry was verified, not assumed.** A 6.8% IoU pass rate is also the
signature of the coordinate bug §5.2 warns about, so it was hand-checked against
raw boxes. On high-IoU rows the model's normalised box matches the annotation to
three decimals — `(0.706, 0.470, 0.824, 0.865)` against `[0.70, 0.47, 0.82, 0.90]`,
IoU 0.851. No broken transform produces that. The distribution has a real tail:
max 0.872, p99 0.774, 32% non-zero.

Three caveats belong with the 2.4%:

1. **Model regions are tiny** — median 0.64% of image area. IoU is harsh for small
   boxes: two hard-hat-sized regions slightly apart score exactly zero. §5.2 calls
   IoU ≥ 0.30 permissive, but that assumed larger regions.
2. **1,430 rows are unscoreable** (`no_annotation_box`) — YES answers on rules the
   image does not violate. **A false alarm has no correct explanation**, so this
   metric structurally cannot penalise the model's largest failure mode. Stage 2's
   59% and Stage 10's 2.4% measure different things.
3. **Zero IoU may mean the model found a real violation nobody boxed.** 47% of the
   corpus is self-flagged "poor info"; separating that from a genuine miss needs
   the §5.5 human audit.

### Pre-registered thresholds

Fixed before any value was computed, and not tuned against results:

| threshold | value | source |
|---|---|---|
| `SIMILARITY_FLOOR` | 0.60 | standard paraphrase floor for MiniLM-class bi-encoders |
| `IOU_FLOOR` | 0.30 | deliberately permissive; the model gives one region, the annotation may mark several |
| `PARSE_RATE_FLOOR` | 0.95 | architecture doc, Stage 2 |
| `LATENCY_CEILING_S` | 25.0 | recalibrated from measured in-run latency — see `runner.py` |

The embedding model is pinned at `1110a243` and verified against §6.3's own worked
pair: paraphrase 0.704, unrelated 0.340. Note the doc's illustrative 0.88 is
higher than this model actually gives; 0.60 still separates them.

All three metric-7 signals are written to the CSV beside the verdict, so
re-thresholding needs no re-run.

## Stage 1 result — the frozen sample

```
pairs                  : 3,882
distinct images        : 2,556   (1,278 hazard + 1,278 matched compliant)
positives per rule     : rule_1 1001   rule_2 84   rule_3 172   rule_4 70
largest strata divergence : 1.5%   (WARNING threshold 5%)
seed                   : 42
sha256                 : af91ed280cd22bfc6cdb27cc6645f448d137bb29b4ec77fc504cddfa3ac166c9
```

The per-rule counts match architecture doc §5.3 exactly. That is the check the
doc itself specifies — "verify per-rule counts match §5.3 exactly; if not, the
scan is buggy" — and it is worth re-running after any change to `rules.py` or
`sampling.py`.

**The sample is both splits, not the test split alone.** The model is evaluated
zero-shot and never trained on any of this data, so there is no contamination to
avoid, and `test` alone would put rule_2 at n=25 and rule_4 at n=24 — below the
pilot's `MIN_N_FLOOR` of 30, making two of the four rules unreportable.

**This file is frozen.** `01_build_sample.py` refuses to overwrite it without
`--force`. Regenerating it invalidates every result computed against it, silently,
because the old result files would still parse.

## Stage 0 result

Qwen2.5-VL-7B-4bit, bf16+sdpa, `max_pixels` = 640 blocks, reason-first format:

```
small-object detection : 89%  (n=9)     threshold 80%   -> GO
sensitivity            : 95%  (n=20)
specificity            : 30%  (n=10)
YES share of replies   : 87%
median latency         : 2.9 s          budget 6.0 s
parse rate             : 100%
peak VRAM              : 7.0 GB
attention available    : True
```

Read the 89% next to the 87%: an always-YES model would score 100% small-object
detection, 0% specificity, 67% overall. This model beats that degenerate baseline
by about six points, so the headline is partly a YES bias. See architecture doc
§12.1 for the full reading, and `results/stage0_feasibility.csv` for per-image rows.

## Settings that are correctness, not tuning

Four defaults were changed after measurement, and each would have produced
confident wrong numbers rather than a crash. Do not revert without re-measuring:

| setting | why |
|---|---|
| `COMPUTE_DTYPE = bfloat16` | fp16 overflows; generation collapses to empty output |
| `ATTENTION_IMPL = sdpa` | the eager kernel injects spurious tokens mid-sentence |
| `MAX_PIXELS = 640` blocks | 1280 spills past the 8 GB card, ~30× slower, no gain in sensitivity |
| `ANSWER` last in the format | with ANSWER first the verdict precedes the reasoning and contradicts it |

---

## Running Stage 2

```bash
cd studies/02-vlm-xai-study
../../pilot/.venv/Scripts/python scripts/01_build_sample.py --seed 42          # once; already done
../../pilot/.venv/Scripts/python scripts/02_baseline_inference.py --model qwen7b-4bit --limit 20 --resume
../../pilot/.venv/Scripts/python scripts/02_baseline_inference.py --model qwen7b-4bit --resume
```

Wall clock for the full pass: **~9.4 h**, measured.

Stage 1 prints a projection of ~3.1 h based on the Stage 0 median of 2.9 s/call.
**That projection is optimistic and should not be planned around.** Stage 0 timed
39 hand-picked gate images; over the full corpus the measured median is **8.84 s**
(p25 3.42, p75 9.48, p90 11.05, max 86.1). Most corpus images reach the
`MAX_PIXELS` cap that the smaller gate images did not, so they carry the full
609-token vision budget rather than less.

Note p90 sits close to the 12.0 s latency abort. A trip several hours in is
plausible; it costs only the time to rerun the command, since every completed row
is already on disk.

**`--resume` is safe on a first run.** With no existing CSV the done-set is empty
and it behaves as a fresh start. Pass it every time — there is then no way to lose
a run by forgetting a flag. After a crash, a reboot, or a Ctrl-C, rerun the exact
same command.

Work is keyed on `(image_id, rule_id)`, never on row position, because images
arrive in dataset streaming order rather than manifest order. Every row is
flushed and `fsync`'d before the next call begins, so a crash loses nothing.

### If it aborts with `FingerprintMismatch`

The results file on disk was produced under a different configuration, and the
error names which field changed. Resuming would concatenate rows answering
different questions into a CSV that still parses — which no downstream metric
could detect. Either start a fresh output file, or pass `--force-resume` if you
are certain the difference does not affect the measurement (it is recorded as
`forced: true` in the state file either way).

The fingerprint covers the manifest sha256, model tier and pinned revision,
`COMPUTE_DTYPE`, `ATTENTION_IMPL`, `MAX_PIXELS`, `GENERATION` and the seed.

### The two guard rails

Checked every 100 rows, over rows completed **in the current process only**, and
inert below 100 such rows — so a resumed run is never aborted by history it
cannot change, and a short `--limit` smoke run never trips them.

| guard | threshold |
|---|---|
| parse rate | `< 95%` → abort |
| median latency | `> 12.0 s` → abort |

The latency ceiling is twice the *declared budget* (6.0 s), not twice the
*observed* Stage 0 median (2.9 s). Twice the observed median is 5.8 s, which sits
below the budget the study already accepted, so it would abort healthy runs. The
failure it exists to catch — Windows silently spilling to shared memory instead
of raising OOM — costs roughly 30x, so it breaches 12.0 s unambiguously.

---

## What can be worked on without a GPU

Most of the interesting logic, as it happens:

```bash
pytest tests -q
```

- **Prompt wording** (`rules.py`). Changing a question and re-running the tests
  takes seconds. The exact phrasing materially affects results, and this is the
  cheapest place to iterate.
- **Reply parsing** (`parsing.py`). Every real model reply that fails to parse is a
  lost data point. Add it to `tests/test_parsing.py` as a case, then make it pass.
- **Sampling and reweighting** (`sampling.py`). Pure arithmetic over metadata.

Model inference needs CUDA — see
[SETUP.md § Which machine runs what](../../docs/setup/SETUP.md#which-machine-runs-what).

---

## Design decisions worth knowing before editing

**Every violated rule is asked.** The pilot kept only the highest-priority
violation and discarded the rest, so a photo with both a PPE breach and an
unguarded trench was only ever tested for PPE.

**Negative controls are context-aware.** The pilot assigned compliant images a rule
by blind rotation, which could ask about excavator proximity in a photo containing
no machinery — a question that cannot be failed, inflating accuracy while measuring
nothing. `rules_to_ask` only draws controls from *applicable* rules.

**The parser is tolerant but records tolerance.** A strict parser silently discards
understandable-but-loosely-formatted replies; a study losing 15% of its data to
formatting is reporting a metric over an unrepresentative subset. So variations are
accepted and flagged `strict=False`, and the non-strict share is reported. A rising
rate means the prompt is degrading.

**Sampling is two strata, not one balanced set.** Three of the seven metrics are
*undefined* on a NO answer — no region to mask, no rationale to grade — so compliant
images cannot contribute to them at all. They measure specificity and the
false-alarm rate instead, and the two strata are recombined by prevalence
reweighting.

Why that last point matters concretely, from `reweight_precision`:

```
sensitivity 0.80, specificity 0.90
  measured on a balanced 50/50 set  →  precision 0.89
  at the true 12.8% prevalence      →  precision 0.54
```

Reporting only the balanced figure overstates deployed precision by 35 points.

---

## Layout

```
src/xai_vlm/
    rules.py          four rules, prompt text, which-rules-to-ask policy
    parsing.py        reply → structured answer, with parse-rate tracking
    sampling.py       two-stratum construction, matching, reweighting
    runner.py         resumption, fingerprinting, guard rails (no torch)
    masking.py        patch-grid alignment
    config.py         settings that must not drift between models or stages
    backends/
        base.py       the Backend protocol and the Answer dataclass
        fake.py       scripted backend, for building the harness GPU-free
        qwen.py       Qwen2.5-VL
    metrics/          the seven metrics (to come)
scripts/              numbered stage runners
tests/                110 tests; 98 need no GPU, 12 need --gpu
results/              output CSVs, including the frozen sample_manifest.csv
```

---

## Next steps

1. **Run Stage 2 to completion** (~3.1 h). Check the negative controls in the
   summary before reading any accuracy figure.
2. **Metrics** (Plan 3), in dependency order: descriptive accuracy → completeness
   → explanation correctness → the rest. Each reuses `runner.py` for resumption.
3. **The §5.5 human audit.** Until it exists, model false alarms cannot be
   separated from annotation misses, and the false-alarm rate stays ambiguous.
