#!/bin/bash -l
#SBATCH --job-name=xai-parity
#SBATCH --account=scholar
#SBATCH --partition=gpu
#SBATCH --constraint=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/scholar/mkamrul/xai/logs/%x-%j.out
set -euo pipefail
X=/scratch/scholar/mkamrul/xai
module load conda/2025.09
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $X/env
export HF_HOME=$X/.hf-home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
echo "== host $(hostname)  commit $(cat $X/repo/COMMIT)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch,transformers,bitsandbytes;print(torch.__version__,transformers.__version__,bitsandbytes.__version__,torch.cuda.get_device_name(0))"
cd $X/repo/studies/02-vlm-xai-study
echo "== GPU tests"
python -m pytest tests -p no:cacheprovider --gpu -m gpu -q -W ignore::UserWarning
echo "== Stage 2 parity run"
python scripts/02_baseline_inference.py --model qwen7b-4bit --manifest results/parity_manifest.csv --out-tag scholar-a40-parity --resume
echo "== comparison"
python cluster/parity/compare_parity.py scholar-a40-parity
echo JOB_DONE
