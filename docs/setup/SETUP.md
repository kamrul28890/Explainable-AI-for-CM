# Machine Setup

How to get this repository running on a new computer, from nothing.

There are two supported machine roles. **Read "Which machine runs what" first** — it
determines which install path you follow, and it will save you from discovering
three hours in that a metric cannot be computed on your hardware.

---

## Which machine runs what

This project is developed across two machines, and the split is deliberate rather
than incidental.

| | **Compute machine** | **Development machine** |
|---|---|---|
| example | Windows + RTX 3070 (8 GB) | MacBook M1 (8 GB unified) |
| runs | all model inference | editing, analysis, figures, writing |
| stages | Stage 0 and Stages 2–8 | Stages 1, 9, 10 and all analysis |
| needs | CUDA | nothing special |

### Why model inference does not run on Apple Silicon

Three independent reasons, each sufficient on its own:

1. **`bitsandbytes` has no Apple Silicon build.** The 4-bit and 8-bit quantisation
   the study specifies is CUDA-only, so the exact model configuration cannot be
   reproduced there.
2. **MLX cannot expose what the metrics need.** MLX *can* run quantised
   Qwen2.5-VL on an M1, but it does not readily expose per-layer attention or
   image-token ablation. Using it would silently drop **Sparsity (metric 2)** and
   the artefact-free masking that **Descriptive Accuracy (metric 1)** relies on.
3. **Speed.** The M1 GPU is roughly 5–8× slower than an RTX 3070 for this
   workload, turning a ~60 hour study into months.

On an **8 GB** M1 there is a fourth: unified memory is shared with the operating
system, so a 3B model in fp16 (~6.6 GB) will swap.

**None of this blocks you.** Results are committed as CSVs, so every statistic,
figure and chapter can be produced on the Mac without touching a model. Only the
inference stages need the CUDA box.

---

## A. Compute machine — Windows or Linux with an NVIDIA GPU

### A1. Prerequisites

- **Python 3.10.x.** Not 3.11+. Florence-2's remote code and the pinned
  `transformers` 4.49.0 are verified against 3.10.11.
- **NVIDIA driver** recent enough for your chosen CUDA build (`nvidia-smi` to check).
- **~20 GB free disk** — 4.5 GB dataset plus model weights.
- **Git**.

### A2. Clone

```bash
git clone https://github.com/kamrul28890/Explainable-AI-for-CM.git
cd Explainable-AI-for-CM
```

### A3. Create the virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

> Never copy a `.venv` between machines or move one to a different folder. It bakes
> in absolute paths — see [Troubleshooting](#troubleshooting).

### A4. Install torch FIRST, from the CUDA index

This step is separate and must come first. If you skip it, pip resolves torch as a
transitive dependency, installs a **CPU-only wheel**, and every run silently falls
back to the CPU at roughly 50× slower with no error.

```bash
pip install torch==2.12.1 torchvision --index-url https://download.pytorch.org/whl/cu130
```

Pick the `cuXXX` tag matching your driver from <https://download.pytorch.org/whl/>.

Verify before continuing — do not proceed if this prints `False`:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.12.1+cu130 True
```

### A5. Install everything else

```bash
pip install -r requirements/base.txt -r requirements/cuda.txt
```

### A6. Install the pilot package (editable)

```bash
pip install -e studies/01-florence2-pilot
```

### A7. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name xai-pilot --display-name "Python (xai-pilot)"
```

In VS Code, select **Python (xai-pilot)** in the kernel picker. The notebook's first
cell verifies the interpreter and fails with a readable message if it is wrong.

### A8. Fetch data and models

```bash
python scripts/download_data.py --what dataset     # ~4.5 GB
python scripts/download_data.py --what models --tier auto
```

`--tier auto` inspects your GPU and picks the model tier. Verify at any time:

```bash
python scripts/download_data.py --what verify
```

Expected: 3,004 + 3,500 + 3,509 = **10,013 rows**.

### A9. Confirm the environment

```bash
python scripts/check_environment.py
```

---

## B. Development machine — macOS (Apple Silicon)

### B1. Prerequisites

- **Python 3.10.x** (`brew install python@3.10`, or pyenv).
- **~6 GB free disk** if you skip the dataset; ~20 GB if you want it locally.
- **Git**.

### B2. Clone and create the environment

```bash
git clone https://github.com/kamrul28890/Explainable-AI-for-CM.git
cd Explainable-AI-for-CM
python3.10 -m venv .venv
source .venv/bin/activate
```

### B3. Install

torch for macOS comes from the normal PyPI index — no `--index-url`, unlike CUDA:

```bash
pip install -r requirements/base.txt -r requirements/mac.txt
pip install -e studies/01-florence2-pilot
python -m ipykernel install --user --name xai-pilot --display-name "Python (xai-pilot)"
```

`bitsandbytes` is skipped automatically here by an environment marker in
`base.txt` — it has no Apple Silicon build.

### B4. Do you need the dataset locally?

Only if you want to inspect images or rebuild figures that render photographs.
Statistics and metric analysis run entirely off the committed CSVs.

```bash
python scripts/download_data.py --what dataset     # optional, ~4.5 GB
```

### B5. Optional — small local smoke tests

MLX can run a quantised 3B model for prompt iteration, though not for producing
study results:

```bash
python scripts/download_data.py --what models --tier mlx-3b-4bit
```

Expect roughly 15–40 s per call on an 8 GB M1, and note that results from this
path are **not comparable** to CUDA runs: different quantisation, no attention
extraction, no token ablation. Use it to check that a prompt parses, never to
generate numbers that go in the report.

---

## Verifying a fresh install

Run these in order. Each should pass before moving to the next.

```bash
# 1. interpreter and libraries
python scripts/check_environment.py

# 2. dataset integrity
python scripts/download_data.py --what verify

# 3. the pilot's own test suite
pytest studies/01-florence2-pilot/tests -q
```

---

## Troubleshooting

### "The Kernel crashed while executing code"

Almost always the **wrong Python interpreter**. A system Python with mismatched
`torch`/`torchvision` aborts the process during import, which surfaces as a kernel
crash with no traceback rather than as an ordinary error.

Check which interpreter the kernel is using:

```bash
python -c "import sys; print(sys.executable)"
```

It must be the `.venv` inside the repository. In VS Code select
**Python (xai-pilot)**. Do **not** work around it by appending `src/` to
`sys.path` — that defers the failure to the point where torch loads, where it
crashes the kernel instead of producing a readable error.

### VS Code hangs on "Connecting to kernel..."

A client/kernel protocol mismatch. ipykernel 7.x rewrote the shell channel as
async and added the subshell protocol; the Jupyter extension 2025.9.1 predates it
and waits forever. `nbconvert` works fine on the same kernel, which makes it look
like a notebook bug.

```bash
pip install "ipykernel<7"
```

This is pinned in `requirements/base.txt`, so it only bites if you install
outside it.

### `ModuleNotFoundError: No module named 'xai_pilot'`

The editable install is missing or stale. It records an **absolute path**, so it
breaks whenever the project folder moves:

```bash
pip install -e studies/01-florence2-pilot
```

### `torch.cuda.is_available()` returns False

A CPU-only wheel was installed. Reinstall from the CUDA index (step A4).

```bash
pip uninstall -y torch torchvision
pip install torch==2.12.1 torchvision --index-url https://download.pytorch.org/whl/cu130
```

### CUDA out of memory

The reference GPU has 8 GB, and typically only ~5.7 GB is free once the desktop
is running.

1. Close other GPU applications (browsers count).
2. Lower `max_pixels` in the study config — for Qwen2.5-VL, **image resolution
   drives memory more than model size does**, because vision tokens and the KV
   cache both scale with it.
3. Drop to a smaller tier: `--tier qwen3b-8bit`.

### Moving the project folder breaks everything

Three things store absolute paths: the editable-install `.pth` file, the Jupyter
kernelspec, and `.vscode/settings.json`. After any move:

```bash
pip install -e studies/01-florence2-pilot
python -m ipykernel install --user --name xai-pilot --display-name "Python (xai-pilot)"
```

`scripts/migrate_pilot_to_studies.sh` does this automatically for the specific
`pilot/` → `studies/01-florence2-pilot/` rename.

### Windows: "Device or resource busy" when moving folders

Windows will not move a directory while an executable inside it is running — and
the venv's `python.exe` lives inside the project. Close VS Code and shut down all
Jupyter kernels first.

---

## What is deliberately *not* in the repository

| item | why | how to get it |
|---|---|---|
| `.venv/` | machine-specific; a Windows/CUDA venv cannot run on macOS | recreate per §A3 / §B2 |
| dataset (~4.5 GB) | far too large for git | `scripts/download_data.py` |
| model weights | several GB, and freely re-downloadable | `scripts/download_data.py` |
| `personal_notes.docx` | private | not published |

Results CSVs **are** committed. That is what lets the development machine do all
analysis without a GPU.
