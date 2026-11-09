#!/bin/bash -l
# One link of a self-continuing job chain on RCAC Scholar.
#
#   sbatch cluster/chain.sbatch <chain>      chain = base | s4 | s5 | s6 | final
#
# Scholar caps a job at 4 h and the battery needs far longer, so each link:
#   1. queues its own successor FIRST (afterany), so a walltime kill still continues;
#   2. runs its chain's steps, skipping any step already marked done;
#   3. on success marks the chain done and cancels the successor;
#      on timeout leaves the successor to resume;
#      on a real failure writes FAILED and cancels the successor -- no retry loop.
#
# `#!/bin/bash -l` is required: without a login shell `module` does not exist.
#SBATCH --account=scholar
#SBATCH --partition=gpu
#SBATCH --constraint=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/scholar/mkamrul/xai/logs/chain-%x-%j.out

set -uo pipefail
CHAIN="${1:?usage: chain.sbatch <base|s4|s5|s6|final>}"
HERE="${XAI_CLUSTER_DIR:-$(cd "$(dirname "$0")" && pwd)}"
source "$HERE/env.sh"
source "$HERE/steps.sh"

mkdir -p "$STATE"
[[ -e "$STATE/$CHAIN.done" ]] && { echo "chain $CHAIN already done"; exit 0; }
[[ -e "$STATE/FAILED" ]] && { echo "FAILED marker present: $(cat "$STATE/FAILED")"; exit 1; }

n=$(( $(cat "$STATE/$CHAIN.count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$STATE/$CHAIN.count"
if (( n > MAX_LINKS )); then
    echo "chain $CHAIN exceeded $MAX_LINKS links" | tee "$STATE/FAILED"; exit 1
fi

NEXT=$(sbatch --parsable --job-name="$CHAIN" --dependency=afterany:"$SLURM_JOB_ID" \
       --export=ALL,XAI_CLUSTER_DIR="$HERE" "$HERE/chain.sbatch" "$CHAIN")
echo "== chain $CHAIN link $n  job $SLURM_JOB_ID  successor $NEXT  host $(hostname)"
echo "== commit $(cat "$REPO/COMMIT")  results $XAI_RESULTS_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

DEADLINE=$(( $(date +%s) + BUDGET_S ))
"chain_$CHAIN"
rc=$?

case $rc in
    0)   touch "$STATE/$CHAIN.done"; scancel "$NEXT"
         echo "== chain $CHAIN DONE"; after_chain "$CHAIN" ;;
    124) echo "== chain $CHAIN out of time; successor $NEXT resumes" ;;
    *)   echo "chain $CHAIN failed rc=$rc in job $SLURM_JOB_ID" | tee "$STATE/FAILED"
         scancel "$NEXT" ;;
esac
exit $rc
