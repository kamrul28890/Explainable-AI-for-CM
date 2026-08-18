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
| `metrics/` — the seven metrics | ⬜ not started (Plan 3) |

108 tests pass: 96 without a GPU, 12 requiring `--gpu`.

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

Projected wall clock for the full pass: **~3.1 h** at the Stage 0 median of
2.9 s/call.

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
tests/                108 tests; 96 need no GPU, 12 need --gpu
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
