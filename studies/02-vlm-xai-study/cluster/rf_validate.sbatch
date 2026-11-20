#!/bin/bash -l
# End-to-end check of the region-frame fix (2026-11-12), before the GPU reruns.
#
#   sbatch cluster/rf_validate.sbatch
#
# Re-asks the first 40 manifest rows with XAI_REGION_FRAME=rescaled, so the
# backend maps REGION itself, and compares each region with the one
# 17_region_frame_fix.py derived offline from the frozen Scholar reply. Where
# the A40 reproduces the frozen reply byte for byte, the two must agree.
#SBATCH --account=scholar
#SBATCH --partition=gpu
#SBATCH --constraint=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=rf_validate
#SBATCH --output=/scratch/scholar/mkamrul/xai/logs/rf_validate-%j.out

set -uo pipefail
RUN_NAME=rf-validate source /scratch/scholar/mkamrul/xai/repo/studies/02-vlm-xai-study/cluster/env.sh
export XAI_REGION_FRAME=rescaled
echo "commit $(cat "$REPO/COMMIT")  results $XAI_RESULTS_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader

env XAI_RESULTS_DIR= XAI_REGION_FRAME= $PY -m pytest tests --gpu -p no:cacheprovider -q -W ignore::UserWarning \
    || { echo "GPU tests failed"; exit 1; }

rm -f "$XAI_RESULTS_DIR"/baseline_qwen7b-4bit.*
$PY scripts/02_baseline_inference.py --model qwen7b-4bit --limit 40 \
    --manifest "$STUDY/results/scholar-a40-rf/sample_manifest.csv" || exit 1

$PY - <<'EOF'
import ast, csv, os
csv.field_size_limit(10**8)
d = os.environ["XAI_RESULTS_DIR"]; s = d.rsplit("/", 1)[0]
rd = lambda p: {(r["image_id"], r["rule_id"]): r for r in csv.DictReader(open(p, encoding="utf-8"))}
new = rd(f"{d}/baseline_qwen7b-4bit.csv")
off = rd(f"{s}/scholar-a40-rf/baseline_qwen7b-4bit.csv")
frz = rd(f"{s}/scholar-a40/baseline_qwen7b-4bit.csv")
box = lambda t: tuple(round(float(v), 1) for v in ast.literal_eval(t)) if t.strip() else None
same_reply = agree = disagree = 0
for k, r in new.items():
    if r["raw_text"] != frz[k]["raw_text"]:
        continue
    same_reply += 1
    a, b = box(r["region"]), box(off[k]["region"])
    ok = a == b or (a and b and max(abs(x - y) for x, y in zip(a, b)) <= 0.11)
    agree += bool(ok); disagree += not ok
    if not ok:
        print("MISMATCH", k, a, b)
print(f"rows {len(new)}  reply identical to frozen {same_reply}  regions agree {agree}  disagree {disagree}")
print("RF_VALIDATE", "PASS" if same_reply and not disagree else "FAIL")
EOF
