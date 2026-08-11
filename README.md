# Explainable AI for Construction-Site Safety

Can we trust the explanation a vision-language model gives when it judges a
construction site unsafe?

Not *"is the answer right?"* — that is ordinary accuracy. The question is whether
the **explanation** holds up. In a domain where a wrong call has physical
consequences and a human supervisor must audit the machine, an unexplainable
system is an undeployable one.

This repository evaluates that question using Prof. Mustafa Abdallah's six-metric
XAI framework, extended with a seventh metric of our own.

---

## Start here

| you want to… | read |
|---|---|
| understand the research design | **[Architecture document](docs/architecture/vlm-xai-study-architecture.md)** — self-contained, assumes no background |
| set up a new machine | [docs/setup/SETUP.md](docs/setup/SETUP.md) |
| regenerate the results | [docs/setup/REPRODUCE.md](docs/setup/REPRODUCE.md) |
| see the pipeline running, end to end | [the walkthrough notebook](studies/01-florence2-pilot/notebooks/00_full_pipeline_walkthrough.ipynb) — outputs committed, no GPU needed |

---

## The two studies

### Study 1 — Florence-2 grounding pilot *(complete, frozen)*

A 163-sample pilot using Florence-2-base-ft. Because that model has **no way to
answer a free-form question**, safety rules had to be answered by a geometric
workaround: ground `"worker"`, ground `"hard hat"`, and call it compliant if the
boxes were close enough.

The pilot's most useful finding is about its own method. Measurement showed the
approach could not support the claims made of it:

- It **never verified the detected object was what was asked for** — the detector
  returns labels, and the code discarded them.
- **Proximity is not wearing.** With a 96-pixel threshold on a 1200-pixel image, a
  hard hat *lying on the ground* marked a worker compliant, and because matching
  was any-to-any, **one helmet satisfied every worker near it**.
- **The rule was narrower than the thing it measured.** `rule_1` covers PPE
  generally, but only `"hard hat"` was ever grounded — **16.9%** of real PPE
  violations were undetectable by construction.

Every metric was therefore measuring *a distance threshold*, not model reasoning.

Results: [`studies/01-florence2-pilot/results/`](studies/01-florence2-pilot/results/)

| metric | overall | n |
|---|---|---|
| descriptive accuracy | 0.362 | 163 |
| visual sparsity (top-5) | 0.108 | 159 |
| stability (answer agreement) | 0.775 | 163 |
| robustness (answer survival) | 0.807 | 652 |
| bounded completeness | 0.436 | 163 |

### Study 2 — VLM XAI study *(designed; implementation starting)*

Replaces the geometric proxy with models that can simply be **asked the question**,
and compares three of them — isolating the effect of model *scale* and model
*family* on explanation quality.

The design adds two things the pilot could not do:

- **Explanation correctness.** The dataset ships a human-written `reason` for every
  violation. The pilot never used it. Grading the model's rationale against it
  measures whether an explanation is *true*, not merely *influential* — and exposes
  the **right-answer-wrong-reason** case, the most dangerous failure mode in
  safety-critical AI and one invisible to every other metric.
- **Artefact-free masking.** Instead of painting a black rectangle no real
  photograph contains, delete the **image tokens** before the language model runs.
  Reporting both side by side measures how much the black-rectangle artefact was
  inflating the result.

Full design: **[docs/architecture/vlm-xai-study-architecture.md](docs/architecture/vlm-xai-study-architecture.md)**

---

## The seven metrics

| # | Metric | Question | Measures |
|---|---|---|---|
| 1 | Descriptive Accuracy | Delete the region it relied on — does the answer change? | faithfulness |
| 2 | Sparsity | Is attention focused, or smeared? | readability |
| 3 | Stability | Ask repeatedly — same answer? | reliability |
| 4 | Efficiency | How long per evaluation? | cost |
| 5 | Robustness | Survives blur, darkness, occlusion, colour loss? | reliability |
| 6 | Bounded Completeness | Was a usable explanation given, and did it matter? | faithfulness |
| 7 | **Explanation Correctness** | Does the stated reason match the human's? | **correctness** |

Faithfulness and correctness are **independent**. An explanation can be faithful
but wrong — the model genuinely used the region it named, and that region was the
wrong one. Cross-tabulating metrics 1 and 7 is the study's central result.

---

## The data

[**ConstructionSite**](https://huggingface.co/datasets/LouisChen15/ConstructionSite) —
10,013 annotated construction-site photographs (7,009 train / 3,004 test).

Nothing here is trained, so the train/test split carries no leakage risk and both
are used. Four safety rules: PPE, fall protection, edge protection, plant proximity.

Two facts drive the entire sampling design:

- **87.2% of images have no violation at all.** Running "the whole dataset" mostly
  means running the majority class.
- **The rare hazards have a hard ceiling.** Plant proximity has **70** positive
  examples in existence. No sampling strategy creates more, and every conclusion
  about that hazard is bounded by it.

Hence a two-stratum design with prevalence reweighting — architecture doc §5.3.

The dataset is **not** in git (~4.5 GB):

```bash
python scripts/download_data.py --what dataset
```

---

## Repository layout

```
docs/
  architecture/     the study design — start here
  setup/            SETUP.md, REPRODUCE.md
studies/
  01-florence2-pilot/   completed pilot, frozen
  02-vlm-xai-study/     the new study
scripts/
  download_data.py      fetch dataset and model weights
  check_environment.py  verify a machine and report what it can run
requirements/
  base.txt  cuda.txt  mac.txt
```

---

## Hardware

Developed across two machines, deliberately:

| role | example | runs |
|---|---|---|
| **compute** | Windows + RTX 3070 (8 GB) | all model inference |
| **development** | MacBook M1 (8 GB) | analysis, figures, writing |

Model inference does not run on Apple Silicon: `bitsandbytes` quantisation is
CUDA-only, MLX cannot expose the attention and image-token access two metrics
require, and the M1 is 5–8× slower. **Results are committed as CSVs**, so all
analysis works without a GPU. Details in
[SETUP.md](docs/setup/SETUP.md#which-machine-runs-what).

---

## Quick start

```bash
git clone https://github.com/kamrul28890/Explainable-AI-for-CM.git
cd Explainable-AI-for-CM
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# CUDA machine — install torch FIRST, from the CUDA index, or you get a
# CPU-only wheel and every run silently falls back to the CPU.
pip install torch==2.12.1 torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements/base.txt -r requirements/cuda.txt

# macOS
pip install -r requirements/base.txt -r requirements/mac.txt

pip install -e studies/01-florence2-pilot
python scripts/check_environment.py
```

`check_environment.py` reports which stages your machine can actually run.

---

## Status

- [x] Study 1 — pilot complete, results frozen and published
- [x] Study 1 — walkthrough notebook with committed outputs
- [x] Study 2 — architecture and methodology designed
- [ ] Study 2 — Stage 0 feasibility gate
- [ ] Study 2 — implementation and full run

---

## Acknowledgements

Six-metric XAI evaluation framework: Prof. Mustafa Abdallah. Dataset:
[LouisChen15/ConstructionSite](https://huggingface.co/datasets/LouisChen15/ConstructionSite).
