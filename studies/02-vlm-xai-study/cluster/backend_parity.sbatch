#!/bin/bash -l
# Roster step 2 (2026-11-13): the shared HFVLMBackend against the frozen
# QwenBackend, both tiers, on the A40 that produced the frozen trees.
#
#   sbatch cluster/backend_parity.sbatch
#
# For 40 rows per tier drawn at random from rows with a region, each backend
# answers the clean image, the black-masked image and the token-ablated image,
# and returns an attention grid. Everything must be identical
# (scripts/20_backend_parity.py). Run under XAI_REGION_FRAME=rescaled, the
# reporting path; the legacy parse is covered by the CPU tests.
#SBATCH --account=scholar
#SBATCH --partition=gpu
#SBATCH --constraint=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --job-name=backend_parity
#SBATCH --output=/scratch/scholar/mkamrul/xai/logs/backend_parity-%j.out

set -uo pipefail
RUN_NAME=backend-parity source /scratch/scholar/mkamrul/xai/repo/studies/02-vlm-xai-study/cluster/env.sh
export XAI_REGION_FRAME=rescaled
echo "commit $(cat "$REPO/COMMIT")  results $XAI_RESULTS_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader

env XAI_RESULTS_DIR= XAI_REGION_FRAME= $PY -m pytest tests --gpu -p no:cacheprovider -q -W ignore::UserWarning \
    || { echo "GPU tests failed"; exit 1; }

rc=0
for tier in qwen7b-4bit qwen7b-bf16; do
    $PY -W ignore scripts/20_backend_parity.py --tier "$tier" --n 40 \
        --baseline "$STUDY/results/scholar-a40-rf/baseline_$tier.csv" || rc=1
done
echo "BACKEND_PARITY_ALL $([[ $rc == 0 ]] && echo PASS || echo FAIL)"
exit $rc
