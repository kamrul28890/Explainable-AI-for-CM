#!/bin/bash -l
# Roster step 3 (design D5): qwen7b-bf16 in environment B on the 96-row parity
# manifest, against the same rows from environment A in the region-frame tree.
#
#   sbatch cluster/envB_parity.sbatch
#
# Only the library differs: same tier, same rows, same A40, same backend class
# (XAI_BACKEND=hf; the shared backend was shown identical to the frozen one in
# environment A, scripts/20_backend_parity.py).
#SBATCH --account=scholar
#SBATCH --partition=gpu
#SBATCH --constraint=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=envB_parity
#SBATCH --output=/scratch/scholar/mkamrul/xai/logs/envB_parity-%j.out

set -uo pipefail
XAI_ENV=/scratch/scholar/mkamrul/xai/envB RUN_NAME=scholar-a40-rf \
    source /scratch/scholar/mkamrul/xai/repo/studies/02-vlm-xai-study/cluster/env.sh
export XAI_REGION_FRAME=rescaled XAI_BACKEND=hf
echo "commit $(cat "$REPO/COMMIT")  results $XAI_RESULTS_DIR"
$PY -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"
nvidia-smi --query-gpu=name --format=csv,noheader

$PY -W ignore scripts/02_baseline_inference.py --model qwen7b-bf16 --resume \
    --manifest "$STUDY/results/parity_manifest.csv" --out-tag envB-parity || exit 1
$PY scripts/21_env_parity.py --a "$XAI_RESULTS_DIR/baseline_qwen7b-bf16.csv" \
    --b "$XAI_RESULTS_DIR/baseline_qwen7b-bf16_envB-parity.csv"
