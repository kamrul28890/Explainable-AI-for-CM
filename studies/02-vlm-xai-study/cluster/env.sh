# Environment for every Scholar job. Sourced, not executed.
X=/scratch/scholar/mkamrul/xai
REPO=$X/repo
STUDY=$REPO/studies/02-vlm-xai-study

# The whole run lives in its own tree. Replies are not reproducible across GPUs
# (runner.backend_device), so Scholar output must never share files with the
# frozen local results in results/.
RUN_NAME=${RUN_NAME:-scholar-a40}
export XAI_RESULTS_DIR=$STUDY/results/$RUN_NAME
STATE=$XAI_RESULTS_DIR/.chain-state

# Stop each link ~25 min before the 4 h wall so the stage exits via SIGINT
# between rows, never by SIGKILL mid-write. 12 links cap any runaway chain.
BUDGET_S=$(( 3*3600 + 35*60 ))
MAX_LINKS=12

# The frozen Stage 1 manifest. A Stage 1 rerun on Scholar must reproduce it.
FROZEN_MANIFEST_SHA=af91ed280cd22bfc6cdb27cc6645f448d137bb29b4ec77fc504cddfa3ac166c9

module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$X/env"
export HF_HOME=$X/.hf-home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY=python
mkdir -p "$XAI_RESULTS_DIR"
cd "$STUDY"
