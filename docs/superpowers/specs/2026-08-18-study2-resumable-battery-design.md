# Study 2 — Resumable Sampling and Baseline Inference — Design

**Date:** 2026-08-18
**Status:** approved
**Covers:** Study 2 Plan 2 — `scripts/01_build_sample.py`, `scripts/02_baseline_inference.py`,
and the new `src/xai_vlm/runner.py`.
**Predecessor:** Plan 1 (backend + Stage 0 gate), complete; Stage 0 verdict GO.

## Goal

Make the study's main measurement launchable and survivable. Stage 1 produces the
frozen sample manifest that every later stage refers to; Stage 2 runs baseline
inference over it and can be killed and resumed without losing work or silently
corrupting the result.

Both scripts exist today as stubs that print `NOT YET IMPLEMENTED` and return 1.

## Why resumability is a correctness requirement, not a convenience

Stage 2 is the input to Stages 3, 4 and 5. A partial or subtly inconsistent
baseline does not fail loudly — it produces a complete-looking CSV that every
downstream metric happily consumes. Two failure modes matter:

1. **Lost work.** A crash at hour eight of an unattended run costs a day.
2. **Silent stitching.** Half a run at one configuration concatenated with half at
   another yields numbers that are wrong in a way no downstream check can detect.

The design treats (2) as the more serious of the two. Plan 1 already produced four
settings that generate confident wrong numbers rather than crashes; this stage adds
a fifth opportunity and closes it explicitly.

## Contract

- **New file:** `src/xai_vlm/runner.py` — the resume machinery, backend-agnostic,
  no CUDA import, exercised entirely by `FakeBackend` in tests.
- **Implemented:** `scripts/01_build_sample.py` (CPU-only),
  `scripts/02_baseline_inference.py` (CUDA).
- **Unchanged:** `rules.py`, `parsing.py`, `sampling.py`, `masking.py`, `backends/`.
  Every helper these stages need is already implemented and tested; this plan wires
  them together and adds nothing to them.
- **Frozen pilot untouched.** Nothing here reads or writes `pilot/results/`.

---

## Stage 1 — `01_build_sample.py`

Runs on any machine. Reads dataset metadata only; no model is loaded.

### Procedure

1. Scan the test split (10,013 records) for violations, scene metadata and
   `quality_of_info`.
2. **Stratum H** — every violation image (expected ~1,278).
3. **Stratum C** — an equal number of compliant images drawn by
   `match_compliant_sample(compliant, target_distribution, n_wanted, seed)`, where
   `target_distribution` is a `Counter` over the hazard stratum's `metadata_key`
   cells (`illumination`, `camera_distance`, `view`).
4. Per image, `rules_to_ask(row, seed)` returns `(rule_id, is_negative_control)`
   pairs — every violated rule, plus one applicable negative control.
5. `expected_answer(rule_id, row)` supplies ground truth per pair. It is **not**
   stored — see below — but every row is asserted against it at write time, so a
   disagreement between the two independent code paths fails the scan loudly.
6. Print `stratification_report(hazard, compliant)` and `summarize_strata(rows)`.
   **A `WARNING` in the report aborts** unless `--allow-imbalance` is passed: a
   compliant pool that could not supply the cells the hazard stratum needs makes
   the H-vs-C comparison a comparison of scene conditions, not of model behaviour.

### Output

`results/sample_manifest.csv`, one row per image-rule pair, columns exactly the
fields of `sampling.SampleRow` — which are also exactly the columns the
architecture document pre-registered in its Stage 1 section:

```
image_id, rule_id, stratum, is_violation, is_negative_control,
quality_of_info, illumination, camera_distance, view
```

**No `expected` column.** `is_violation` is per *rule*, not per image — the
architecture document's worked example gives image `0000102` two rows, `rule_1`
with `is_violation=True` and `rule_3` with `is_violation=False`. So ground truth is
already present: `expected = "YES" if is_violation else "NO"`. Storing it a second
time would add a field that can disagree with its own source, and would drift the
file away from the schema the study pre-registered.

Row count is not known in advance and is not asserted. Stratum H images contribute
one row per violated rule plus one control; stratum C images contribute one control
row each. The script prints the realised count and the projected Stage 2 wall clock
at the Stage 0 median of 2.9 s/call.

### Freezing

The manifest defines the sample for every later stage. Regenerating it mid-study
invalidates everything already computed, because Stage 4's rows would refer to
different images than Stage 2's — silently, since both files would still parse.

So: **the script refuses to overwrite an existing manifest.** `--force` is required,
and prints the old and new sha256 with a warning naming the result files that would
be invalidated. `--seed` defaults to 42; the same seed must reproduce the same
manifest sha256, and a test asserts this.

---

## Stage 2 — `02_baseline_inference.py` and `runner.py`

### Division of labour

`runner.py` owns resumption, checkpointing, fingerprinting and the guard rails. It
knows nothing about vision models. The script owns loading the manifest, streaming
images, and calling the backend. Stages 3–5 will reuse `runner.py` unchanged; that
is why it is a module and not script-local code.

### Checkpoint files

```
results/baseline_{model}.csv          append-only, flush + fsync after every row
results/baseline_{model}.state.json   run fingerprint
```

### The fingerprint

`state.json` records, at first write:

| field | source |
|---|---|
| `manifest_sha256` | sha256 of `sample_manifest.csv` |
| `model` / `revision` | tier name and the pinned revision from `requirements/models.lock` |
| `compute_dtype` | `config.COMPUTE_DTYPE` |
| `attention_impl` | `config.ATTENTION_IMPL` |
| `max_pixels` | `config.MAX_PIXELS` |
| `generation` | `config.GENERATION` |
| `seed` | CLI seed |

On `--resume`, every field is compared against the live configuration and **any
mismatch aborts** with a diff. This is the mechanism that makes silent stitching
impossible. `--force-resume` overrides it and stamps `forced: true` into the state
file so the provenance survives into the results.

### Resume identity

Done work is keyed on `(image_id, rule_id)` — the natural key of a manifest row —
never on row position. Images arrive in dataset streaming order rather than
manifest order, so a position-based cursor would desync the first time iteration
changed. On resume the runner reads the existing CSV, builds the done-set, and
skips those pairs without loading the model for them.

A resumed run therefore re-streams the dataset from the start. That cost is decode
only, no GPU, and is accepted in exchange for a resume rule that cannot go wrong.

### Memory

Stage 0 accumulated every gate image in a dict, which was fine for 39 images. At
~2,556 it is not. The runner holds one image at a time and releases it after the
call.

### Per-call error handling

```
try:    answer = backend.answer(image, rule_id)
except: torch.cuda.empty_cache(); retry once
        still failing -> append row with error=<repr>, continue
```

`BackendUnavailable` is not retried — it is an environment fault, and it aborts
immediately with its reason rather than writing thousands of error rows.

### Guard rails — checked every 100 rows, pre-registered here

| guard | threshold | why |
|---|---|---|
| parse rate | `< 0.95` → ABORT | `02`'s own docstring rule, enforced during the run rather than after it. A prompt that degrades at hour two otherwise burns the remaining hours producing unusable rows. |
| median latency | `> 12.0 s`, i.e. `2 x LATENCY_BUDGET_S` → ABORT | The silent-spill signature. Windows' driver trades OOM for a ~30x slowdown without raising, so wall clock is the only available detector, and it is invisible unless something watches it. |

Both thresholds are fixed here, before any Stage 2 numbers are seen, for the same
reason the Stage 0 gate thresholds were.

The latency guard is pinned to twice the **pre-registered budget** (6.0 s), not
twice the **observed** Stage 0 median (2.9 s). Twice the observed median is 5.8 s,
which sits below the budget the study already declared acceptable, so normal
variation across a larger and more varied image set could abort a healthy run. A
spill costs roughly 30x, so it breaches 12.0 s just as unambiguously.

Guards evaluate over rows completed **in the current process only**, and do not
activate until 100 such rows exist. A resumed run is therefore never aborted by
history it cannot change, and a short `--limit` smoke run never trips them.

### Progress reporting

Every 100 rows the runner prints rows done / total, elapsed and projected
remaining, running parse rate, and running **sensitivity, specificity and YES
share**. Stage 0 showed the model answers YES to 87% of images; keeping that figure
live means a repeat of the YES-bias pattern is visible in the first minutes rather
than at the end.

### End of run

Print `ParseStats.summary()`, then the negative-control check **first**, before any
accuracy figure. Per `02`'s docstring: if the model answers YES to hazards known to
be absent, its accuracy on violations is meaningless. The summary orders itself to
make that unavoidable rather than optional.

### Output columns

```
image_id, rule_id, stratum, is_negative_control, expected,
verdict, region, rationale, confidence, raw_text,
latency_ms, parse_strict, parse_error, error
```

The model-reply fields map directly from `backends.base.Answer`; none of them are
derived at write time, so the CSV stays a faithful record of what the model said.
`expected` is carried here (unlike in the manifest) because a results file should
be readable on its own, and it is computed from the manifest's `is_violation` at
read time rather than stored twice on disk.

`parse_error` and `error` are distinct and must not be merged. `parse_error` comes
from `Answer` and means the model replied but the reply could not be parsed —
a data point about the prompt. `error` means the call itself raised and there is no
reply at all — a data point about the machine. Collapsing them would let
infrastructure faults masquerade as prompt degradation in the parse rate.

### CLI surface

```
01_build_sample.py  [--seed 42] [--out results/sample_manifest.csv]
                    [--force] [--allow-imbalance]

02_baseline_inference.py  --model qwen7b-4bit [--resume] [--limit N]
                          [--manifest results/sample_manifest.csv]
                          [--checkpoint-every 100] [--force-resume]
```

`--resume` is safe to pass on a first run: with no existing CSV the done-set is
empty and it behaves as a fresh start. This is deliberate, so the same command can
be used every time and there is no way to lose a run by forgetting a flag.

---

## Testing

### `runner.py` — TDD, no GPU

- **Crash and resume.** Run with `FakeBackend`, raise after N rows, resume, assert
  the union of both passes is exactly the manifest — no duplicates, no gaps.
- **Resume is idempotent.** Resuming a complete run does zero calls and exits 0.
- **Fingerprint mismatch aborts.** Change `max_pixels` between passes; assert abort
  and assert the CSV is left untouched.
- **Guards trip.** A `FakeBackend` scripted to emit unparseable replies must abort
  at the first 100-row checkpoint, not at the end.
- **Append durability.** Assert every row is on disk before the next call begins.

### Stage 1

- Determinism: same seed → identical manifest sha256.
- Overwrite refusal: second run without `--force` exits non-zero, file unchanged.
- Every stratum-C row has `is_negative_control=True`; every manifest rule is
  applicable to its image.

### GPU validation

Per the project's standing rule, unit tests are not sufficient on their own. A real
`--limit 20` smoke run on the CUDA machine precedes the full launch, and the
before/after numbers are reported.

## Out of scope

- **Auto-restart after reboot.** Recovery is manual: rerun the same command with
  `--resume`.
- **The specificity-30% finding.** That is the §5.5 human-audit question. This
  stage makes it *visible* live; it does not address it.
- **Stages 3–5.** They consume `runner.py` but are Plan 3.

## Risks

1. **The manifest is larger or smaller than projected.** Row count depends on the
   realised violation distribution. Mitigated by printing the count and projected
   wall clock at the end of Stage 1, before any GPU time is spent.
2. **Re-streaming on resume is slow for many small resumes.** Acceptable; decode is
   CPU-bound and cheap next to a 2.9 s model call. If a run is resumed dozens of
   times this becomes the bottleneck, and a position hint can be added later
   without changing the key-based correctness rule.
3. **`fsync` per row costs throughput.** At 2.9 s/call the write is noise. Measured
   in the smoke run rather than assumed.
