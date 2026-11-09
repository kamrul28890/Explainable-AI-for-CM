# Running Study 2 on RCAC Scholar

From 2026-11-09 the model-inference stages run on Purdue's Scholar cluster
(A40, 48 GB) instead of the local RTX 3070 (8 GB). This folder holds the job
scripts. Everything here assumes the layout below and user `mkamrul`.

```
/scratch/scholar/mkamrul/xai/
    repo/     git archive of a commit, plus COMMIT naming it
    env/      conda env: Python 3.10.11, packages pinned to the local machine
    logs/     one log per job
```

## Run the battery

```bash
ssh scholar
cd /scratch/scholar/mkamrul/xai/repo/studies/02-vlm-xai-study
sbatch --job-name=base cluster/chain.sbatch base
```

That single command runs everything that exists, unattended:

| chain | steps | runs after |
|---|---|---|
| `base` | all tests incl. GPU · Stage 0 · Stage 1 (must reproduce sha `af91ed28…`) · Stage 2 · Stages 3, 7 | — |
| `s4` `s5` `s6` | Stage 4 · Stage 5 · Stage 6, one A40 each, in parallel | `base` |
| `final` | Stages 9, 10 · Stage 11 report (text and LaTeX) | all of `s4` `s5` `s6` |

Stage 8 (robustness) is not implemented and is not run.

Output goes to `results/scholar-a40/` via `XAI_RESULTS_DIR` -- **never** into
`results/`, which holds the frozen local run. Progress markers live in
`results/scholar-a40/.chain-state/`; `COMPLETE` appears there at the end, and
`FAILED` (with the job id) if any step fails for a reason other than time.

## Why a chain

Scholar caps every job at **4 hours**. Each link queues its successor before it
starts work, runs its steps, and gives each stage a SIGINT at 3 h 35 min, so it
stops between rows and closes its CSV rather than being killed mid-write.
The successor resumes: finished steps are skipped by marker, and an unfinished
stage resumes row by row with `--resume`. A link that fails for any other reason
cancels its successor -- there is no retry loop -- and no chain may exceed 12
links.

## Things that bite on Scholar

- **`#!/bin/bash -l`** on every job script, or `module` does not exist.
- **Ask for `--constraint=A40`** (or A30). The V100s have no bf16 hardware, and
  this study requires bf16 -- fp16 makes the model emit empty replies.
- **The dataset is gated** on Hugging Face and the cluster has no token. It was
  uploaded from the local machine and verified by sha256.
- **Replies are not reproducible across GPUs.** The parity run below is why the
  device is in every stage fingerprint and why Scholar has its own results tree.

## Parity with the local machine (2026-11-09)

`parity/` reruns Stage 2 on 96 stratified rows of the frozen manifest and
compares row by row. Same weights, code, data and pinned libraries:

```
verdict identical      93/96 (96.9%)
reply identical        41/96 (42.7%)
region identical       60/96 (62.5%)   differing: median IoU 0.83
sensitivity  81.2% vs 84.4% frozen     specificity 39.1% vs 39.1%
median latency  2.43 s vs 10.82 s frozen
```

Greedy decoding reproduces exactly on one machine (a GPU test checks this) and
not across machines.
