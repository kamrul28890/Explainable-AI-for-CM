# Reproducing This Work

Exact steps to regenerate every result. Assumes [SETUP.md](SETUP.md) is complete.

Two independent bodies of work live here:

- **Study 1 — the Florence-2 pilot.** Complete and frozen. Reproducing it should
  give byte-identical CSVs.
- **Study 2 — the VLM XAI study.** In progress. Designed in
  [the architecture document](../architecture/vlm-xai-study-architecture.md).

---

## Part 1 — Reproducing the frozen pilot

### What "frozen" means

The pilot's published numbers are a scientific record. Every behavioural change
made after publication sits behind a **default-off configuration flag**, so running
with defaults reproduces the original results exactly. The notebook asserts this
and fails loudly if a default has drifted.

The five flags, with their frozen values:

| flag | frozen value | what it controls |
|---|---|---|
| `REGION_RANKING` | `"area"` | which candidate region is masked first |
| `LABELING` | `"priority"` | how many labels a multi-hazard image gets |
| `COMPLIANT_ASSIGNMENT` | `"round_robin"` | which rule a violation-free image is tested against |
| `DECODING` | `"beam"` | text-generation strategy (`num_beams=3`) |
| `REPORT_WORKER_LOSS_CORRECTED` | `False` | raw flip rates only |

### Run it

Requires the **compute machine** (CUDA). About 1.5 hours end to end.

```bash
cd studies/01-florence2-pilot

python scripts/01_check_environment.py
python scripts/02_select_samples.py
python scripts/03_run_baseline_inference.py
python scripts/04_extract_regions.py
python scripts/05_descriptive_accuracy.py
python scripts/06_visual_sparsity.py
python scripts/07_stability_test.py
python scripts/08_efficiency_test.py
python scripts/09_robustness_test.py
python scripts/10_bounded_completeness.py
python scripts/11_make_figures.py
python scripts/13_statistics_report.py
```

Stages are strictly sequential — each reads the previous one's CSV.

### Verify against the published record

```bash
python -c "
import pandas as pd
df = pd.read_csv('results/pilot_metric_summary.csv')
print(df[['metric','overall','n']].to_string(index=False))
"
```

Expected:

| metric | overall | n |
|---|---|---|
| `descriptive_accuracy` | 0.3620 | 163 |
| `visual_sparsity_top5` | 0.1080 | 159 |
| `stability_answer_agreement` | 0.7751 | 163 |
| `robustness_level1_answer_survival` | 0.8067 | 652 |
| `bounded_completeness` | 0.4356 | 163 |

**Stability will not match exactly.** It uses sampling-based decoding
(`do_sample=True`) precisely so reruns *can* disagree — that is the point of the
metric. Expect it within roughly ±0.05. Everything else is deterministic and
should match to four decimal places.

### Exploring the pilot without running it

```bash
jupyter notebook studies/01-florence2-pilot/notebooks/00_full_pipeline_walkthrough.ipynb
```

Outputs are committed, so it reads top to bottom with no GPU. To execute it, use
**Run All** with the `Python (xai-pilot)` kernel — cells share one session, so
running a middle cell on a fresh kernel fails with `NameError`.

### Non-default modes

Each stage accepts `--region-ranking`. Non-default runs write to mode-suffixed
CSVs, so the frozen record is never overwritten:

```bash
python scripts/04_extract_regions.py --region-ranking rule_aware
# writes results/region_extraction_rule_aware.csv
```

---

## Part 2 — Running the VLM XAI study

Read [the architecture document](../architecture/vlm-xai-study-architecture.md)
first. It defines the stages, outputs and interpretation rules. Below is the
operational sequence only.

### Stage 0 is a gate, not a formality

```bash
cd studies/02-vlm-xai-study
python scripts/00_feasibility_gate.py --tier auto
```

**Stop if it fails.** The stop rules are fixed in advance (architecture doc §8,
Stage 0), which is deliberate: writing them beforehand prevents rationalising a
bad result afterwards. The decisive check is whether the quantised model can
still detect small hard hats — if it cannot, `rule_1` is invalid and so is most
of the study.

Stage 0 also measures real seconds-per-call. **The ~60 h budget assumes ~4 s.** If
the measurement is materially higher, rescope before committing, cutting in this
order: robustness subset → stability subset → third model. Never Stage 4 — it is
the primary faithfulness measurement.

### The remaining stages

```bash
python scripts/01_build_sample.py          # writes sample_manifest.csv -- then FREEZE it
python scripts/02_baseline_inference.py --model qwen7b-4bit
python scripts/03_extract_regions.py       --model qwen7b-4bit
python scripts/04_descriptive_accuracy.py  --model qwen7b-4bit
python scripts/05_sparsity.py              --model qwen7b-4bit
python scripts/06_stability.py             --model qwen7b-4bit
python scripts/07_efficiency.py            --model qwen7b-4bit
python scripts/08_robustness.py            --model qwen7b-4bit
python scripts/09_completeness.py          --model qwen7b-4bit
python scripts/10_explanation_correctness.py --model qwen7b-4bit
```

Then repeat stages 02, 04 and 10 for the comparison models (`qwen3b-8bit`,
`internvl4b`) — the core battery the headline comparison needs.

### Why the manifest must be frozen

`sample_manifest.csv` defines the sample. Every later stage keys off it.
Regenerating it mid-study silently invalidates everything already computed,
because stage 4's results would refer to different images than stage 2's.

Generate once, commit it, and do not rerun stage 01 unless you intend to discard
all downstream results.

### Long runs

Every inference stage checkpoints every 100 samples and resumes automatically:

```bash
python scripts/02_baseline_inference.py --model qwen7b-4bit --resume
```

At ~60 hours total, a crash at hour eight without resumability costs a full day.

---

## Which machine to use

| task | machine |
|---|---|
| Stage 0, and all model-inference stages | **compute** (CUDA) |
| Stage 1 (sample construction) | either |
| Statistics, figures, tables, writing | either — analysis reads committed CSVs |
| Notebook authoring and review | either |

Results are committed as CSVs specifically so analysis never needs a GPU. See
[SETUP.md](SETUP.md) § "Which machine runs what".

---

## Determinism and what will not match exactly

| source | reproducible? | notes |
|---|---|---|
| sample selection | yes | seeded; `SEED = 42` |
| Florence-2 baseline, masking, robustness | yes | beam search is deterministic |
| stability | **no, by design** | sampling decoding; expect ±0.05 |
| VLM generation | mostly | greedy/beam deterministic; sampling stages are not |
| timings | no | hardware-dependent — that is the measurement |
| bootstrap CIs | yes | seeded resampling |

Any stochastic stage records its seed in the output CSV. Sampling-based stages run
across five seeds and report mean ± interval rather than a single lucky run.

---

## If numbers do not match

1. **Check the config flags** — the notebook's assertion catches drift, but
   scripts do not. Print them and compare against the table above.
2. **Check library versions** — `python scripts/check_environment.py`.
   `transformers` must be 4.49.0.
3. **Check the dataset** — `python scripts/download_data.py --what verify`.
   Expected 10,013 rows total.
4. **Check the sample** — a different `SEED` or `n_per_class` produces a different
   manifest and therefore different numbers throughout.
5. **Is it stability?** That metric is intentionally non-deterministic.
