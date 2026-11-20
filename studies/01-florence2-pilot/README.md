# XAI Pilot — Florence-2 on ConstructionSite 10k

Technical pilot adapting Professor Mustafa Abdallah's six-metric XAI evaluation framework
(Descriptive Accuracy, Sparsity, Stability, Efficiency, Robustness, Bounded Completeness)
to a Vision-Language Model doing construction-site safety judgments.

Full writeup: [`report/pilot_report_draft.md`](report/pilot_report_draft.md) and the
day-by-day LaTeX chapters in [`report/chapters/`](report/chapters/).

## Setup

`torch` must be installed first, pinned to the CUDA build matching your GPU — installing
`requirements.txt` first can silently pull a CPU-only `torch` wheel and overwrite it:

```bash
pip install torch==2.12.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e .
```

(Check https://download.pytorch.org/whl/ for the `cuXXX` tag matching your installed CUDA
driver if `cu130` doesn't apply.)

The pinned `transformers==4.49.0` is deliberate: `transformers>=5.0` removed legacy
generation attributes that Florence-2's pinned remote code still relies on.

## Data

Pilot sample selection lives in `data/pilot_samples.csv` (image IDs, rule labels, balanced
draw of 50/class where the dataset allowed it) and `data/safety_prompts.json` (per-rule
grounding query phrases). The underlying images come from
[`LouisChen15/ConstructionSite`](https://huggingface.co/datasets/LouisChen15/ConstructionSite)
on Hugging Face (gated, `cc-by-nc-4.0`) and are not stored in this repo.

## Pipeline

Run in order from `pilot/`:

```bash
python scripts/01_check_environment.py      # verify GPU + pinned package versions
python scripts/02_select_samples.py          # balanced sample draw -> data/pilot_samples.csv
python scripts/03_run_baseline_inference.py  # Florence-2 grounding inference -> results/baseline_predictions.csv
python scripts/04_extract_regions.py         # standardize explanation regions
python scripts/05_descriptive_accuracy.py    # mask top-ranked region, check answer flip
python scripts/06_visual_sparsity.py         # cross-attention heatmap concentration
python scripts/07_stability_test.py          # sampled reruns vs. deterministic reference
python scripts/08_efficiency_test.py         # GPU-time extrapolation to n=1000
python scripts/09_robustness_test.py         # perturbations + reworded prompts
python scripts/10_bounded_completeness.py    # is the top-ranked region actually necessary?
python scripts/11_make_figures.py            # roll up CSVs into summary figures
```

Tests: `pytest tests/`

## Layout

| Folder | Contents |
|---|---|
| `src/xai_pilot/` | Library code: model wrapper, attribution, regions, perturbations, metrics. |
| `scripts/` | Numbered pipeline stages (run in order, see above). |
| `tests/` | Unit tests for `src/xai_pilot/`. |
| `data/` | Pilot sample selection and rule-query metadata (not the raw image dataset). |
| `results/` | Per-metric CSVs, day-by-day findings notes, and generated figures. |
| `report/` | Pilot report draft, summaries, and the LaTeX chapter source. |
