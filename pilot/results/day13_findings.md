# Day 13 — From-Scratch Reproducibility Rerun

Per the plan's Day 13 verification criterion ("the from-scratch 20-sample rerun completing
without manual intervention is the actual test"): exported the exact committed `pilot/`
tree at `HEAD` (`git archive HEAD pilot`) into a throwaway directory outside the repo
(`D:\xai_pilot_repro_check\pilot`, deleted after this check), created a brand-new venv
there, ran `pip install` from a clean slate, and ran `scripts/01` through `scripts/11` in
order against a fresh 20-sample subset (`SAMPLES_PER_CLASS` patched from 50 to 5 in that
throwaway copy only — 5/class × 4 classes = 20, never touching the real `pilot/data` or
`pilot/results`).

## Result: full pipeline reproduces from scratch, exit 0 on every script

| step | exit code | notes |
|---|---|---|
| `pytest tests/ -v` | 0 | 61/61 passed, same as the main venv |
| `01_check_environment.py` | 0 | GPU confirmed (`cuda:0`, fp16), gated dataset access worked using the existing user-level HF token (no extra login needed in the new venv — `huggingface_hub`'s token cache lives at `~/.cache/huggingface`, not inside the venv) |
| `02_select_samples.py` | 0 | wrote exactly 20 rows, 5/class, no underfilled-class warning |
| `03_run_baseline_inference.py` | 0 | 20/20 |
| `04_extract_regions.py` | 0 | 20/20, grid-fallback 5.0% (1/20) |
| `05_descriptive_accuracy.py` | 0 | 20/20 — see network-blip note below |
| `06_visual_sparsity.py` | 0 | 20/20 |
| `07_stability_test.py` | 0 | 20/20 |
| `08_efficiency_test.py` | 0 | 20/20 + 15-sample calibration |
| `09_robustness_test.py` | 0 | 20/20 + patch-stretch (see note below) |
| `10_bounded_completeness.py` | 0 | 20/20 |
| `11_make_figures.py` | 0 | wrote summary CSV + chart |

**No manual intervention was needed at any step** — the install order (`torch` from the
pinned CUDA index first, then `requirements.txt`, then `pip install -e .`) worked exactly
as `requirements.txt`'s own comment describes, and the exact-version pin (`torch==2.12.1`)
meant the second install step didn't try to replace the already-installed CUDA build with
a CPU one (verified directly: `torch.__version__` was `2.12.1+cu130` and
`torch.cuda.is_available()` was `True` both before and after `pip install -r
requirements.txt`).

## Two real things this rerun surfaced

**1. A genuine bug in `11_make_figures.py`, found and fixed.** The chart's legend and
title hardcoded `"n=50"` / `"n=13"` and `"163-sample pilot"` — correct for the main pilot,
silently wrong for any other sample size (this 20-sample run would have shown a chart
mislabeled "163-sample pilot" with "n=50" legends next to bars actually computed from 5
samples each). Fixed to read `primary_class` value counts and `len(da)` directly from the
already-loaded `descriptive_accuracy.csv` instead of hardcoding either number. Re-ran
`11_make_figures.py` against the real 163-sample data after the fix — output is
byte-identical in its numbers (the fix only touches chart labels, no computation), and the
chart was visually re-confirmed correct (`results/figures/summary/metric_summary_by_class.png`,
n=50/50/50/13, "163-sample pilot" — unchanged because that's still the correct real n).

**2. A transient network error during `05_descriptive_accuracy.py`'s dataset streaming,
auto-recovered.** `IncompleteRead`/`ProtocolError` while streaming the test-split parquet
from the Hub, retried automatically by `huggingface_hub`'s built-in retry logic (3
retries, exponential backoff) and succeeded. Not a code bug and not something this rerun
needed to fix — flagged here because it's a real source of pipeline flakiness worth being
aware of if a future full-dataset run hits it more than once (a `--retries`/timeout
config tweak would be the fix, not urgent at this scale).

## 20-sample numbers vs. the real 163-sample pilot — directionally consistent, as expected

| metric | 163-sample (real) | 20-sample (repro check) |
|---|---|---|
| descriptive_accuracy | 36.2% | 30.0% |
| visual_sparsity_top5 | 0.108 | 0.092 |
| stability_answer_agreement | 77.5% | 93.3% |
| robustness_level1_answer_survival | 80.7% | 83.8% |
| bounded_completeness | 43.6% | 35.0% |
| efficiency @ n=1000 (GPU-hours) | ~0.50 | ~0.54 |

Four of six land close to the real pilot's numbers; `stability_answer_agreement` is
notably higher (93.3% vs 77.5%) at n=20 — expected sampling noise at this size (20 samples
across 4 classes is a small enough draw that one or two rule_2/rule_3 samples landing on
the "stable" side of the distribution swings the mean substantially), not a discrepancy
that calls the pipeline's correctness into question. This rerun's purpose was reproducing
the *pipeline*, not reproducing the *exact numbers* at 8x fewer samples — and the rule-level
ordering that Days 6/7/10 already explained mechanistically (excavator-class objects
score lowest on sparsity, rule_4 highest on stability/robustness) holds in both runs,
which is the more meaningful reproducibility signal than exact percentage match.

## Cleanup

`D:\xai_pilot_repro_check\` (the throwaway venv + 20-sample data/results) was deleted
after this check completed — it was scratch space for this verification only, not a
tracked artifact.

## Verification performed

- All 11 scripts' exit codes confirmed 0 directly from each background task's result, not assumed from "it looked like it printed output."
- `pytest tests/ -v` passed 61/61 in the fresh venv before any GPU script ran, confirming the editable install resolved correctly.
- Spot-checked `torch.cuda.is_available()` before and after the two-step `requirements.txt` install to confirm the documented install-order risk didn't silently regress this rerun's GPU usage.
- Visually re-inspected the regenerated `metric_summary_by_class.png` after the `11_make_figures.py` fix to confirm the real 163-sample chart is unchanged.
