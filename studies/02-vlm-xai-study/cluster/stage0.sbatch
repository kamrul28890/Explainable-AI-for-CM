#!/bin/bash -l
# Stage 0 alone, for sizing a model before committing GPU-days to it.
#
#   sbatch --gres=gpu:2 --export=ALL,MODEL=qwen32b-bf16,NGPU=2,RUN_NAME=scholar-a40-rf \
#       cluster/stage0.sbatch
#
# Writes stage0_feasibility_<model>.csv into the run tree and prints the gate:
# small-object detection, sensitivity, specificity, latency, peak VRAM. It does
# not mark the chain's stage0 step done; the base chain reruns it.
#SBATCH --account=scholar
#SBATCH --partition=gpu
#SBATCH --constraint=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --job-name=stage0
#SBATCH --output=/scratch/scholar/mkamrul/xai/logs/stage0-%x-%j.out

set -uo pipefail
source /scratch/scholar/mkamrul/xai/repo/studies/02-vlm-xai-study/cluster/env.sh
echo "commit $(cat "$REPO/COMMIT")  model $MODEL  NGPU $NGPU  results $XAI_RESULTS_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -W ignore scripts/00_feasibility_gate.py --tier "$MODEL" \
    --out "$XAI_RESULTS_DIR/stage0_feasibility_$MODEL.sizing.csv"
