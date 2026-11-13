#!/bin/bash -l
# Environment B (roster design D5), built on the login node: no GPU needed.
#
#   nohup bash -l cluster/setup_envB.sh >/scratch/scholar/mkamrul/xai/logs/setup_envB.log 2>&1 &
#
# A CLONE of environment A with only the model libraries moved, so the parity
# run of qwen7b-bf16 in B measures the library change and nothing else. torch is
# held at A's version by a constraint. Versions are at least two weeks old
# (transformers 5.20.2: 2026-10-19; accelerate 1.15.0: 2026-09-09). huggingface-hub
# stays on 1.x (1.33.0, 2026-09-24): transformers allows >=1.31,<3 and the
# datasets build in A needs <2.
set -euo pipefail
X=/scratch/scholar/mkamrul/xai
module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ ! -d $X/envB ]]; then
    conda create -y -p "$X/envB" --clone "$X/env"
fi
conda activate "$X/envB"
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "torch==$TORCH" > "$X/envB.constraints.txt"
pip install -c "$X/envB.constraints.txt" transformers==5.20.2 accelerate==1.15.0 \
    huggingface-hub==1.33.0
pip check
python - <<'PY'
import torch, transformers, accelerate, huggingface_hub
print("ENVB", "torch", torch.__version__, "transformers", transformers.__version__,
      "accelerate", accelerate.__version__, "hub", huggingface_hub.__version__)
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: F401
import datasets  # noqa: F401  (the image loader)
import qwen_vl_utils  # noqa: F401
print("ENVB_IMPORTS_OK")
PY
pip freeze > "$X/envB.freeze.txt"
echo "ENVB_DONE $(date -Is)"
