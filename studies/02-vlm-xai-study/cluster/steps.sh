# The battery, split into chains. Sourced by chain.sbatch.
#
#   base  : GPU tests, Stage 0, Stage 1 (verified against the frozen sha),
#           Stage 2, then the CPU stages 3 and 7 that only need Stage 2
#   s4 s5 s6 : Stages 4, 5, 6 -- independent of each other, so one A40 each
#   final : Stages 9, 10 and 11, once s4, s5 and s6 are all done
#   comp  : on its own -- the 7,457 compliant images the sample left out
#           (Stage 1 --complement, sha-verified), Stage 2 on them, report
#
# Stage 8 (robustness) is not implemented and is not run.

# step NAME CMD... -- run once. Marker on success; 124 when out of time.
step() {
    local name=$1; shift
    if [[ -e "$STATE/step.$name" ]]; then echo "-- $name: done earlier"; return 0; fi
    local left=$(( DEADLINE - $(date +%s) ))
    if (( left < 300 )); then echo "-- $name: under 5 min left, deferring"; return 124; fi
    echo "-- $name: start $(date -Is)  (budget ${left}s)"
    local rc
    if declare -F "$1" >/dev/null; then
        "$@"; rc=$?          # a shell function: quick, and timeout cannot run it
    else
        # SIGINT, not SIGTERM: the stage stops between rows and closes its CSV.
        timeout --signal=INT --kill-after=120 "$left" "$@"; rc=$?
    fi
    echo "-- $name: rc=$rc  $(date -Is)"
    if (( rc == 0 )); then touch "$STATE/step.$name"; return 0; fi
    return $rc               # 124 = out of time; anything else is a failure
}

# check_sha FILE SHA LABEL -- a rebuilt manifest must be the one results cite.
check_sha() {
    local got
    got=$(sha256sum "$1" | cut -d' ' -f1)
    echo "$3 sha256 $got"
    [[ "$got" == "$2" ]] || { echo "$3 did NOT reproduce the frozen sha ($2)"; return 1; }
}
verify_manifest()   { check_sha "$XAI_RESULTS_DIR/sample_manifest.csv" "$FROZEN_MANIFEST_SHA" manifest; }
verify_complement() { check_sha "$XAI_RESULTS_DIR/complement_manifest.csv" "$COMPLEMENT_SHA" complement; }

chain_base() {
    step cpu_and_gpu_tests env XAI_RESULTS_DIR= $PY -m pytest tests --gpu -p no:cacheprovider -q -W ignore::UserWarning || return $?
    step stage0 $PY scripts/00_feasibility_gate.py --tier "$MODEL" \
        --out "$XAI_RESULTS_DIR/stage0_feasibility$SUFFIX.csv"             || return $?
    # The sample is model-independent. Only the primary tier builds it; every
    # other tier must find the same frozen manifest already in the tree.
    if [[ -z $SUFFIX ]]; then
        step stage1 $PY scripts/01_build_sample.py --seed 42                || return $?
    fi
    step stage1_verify verify_manifest                                      || return $?
    step stage2 $PY scripts/02_baseline_inference.py --model "$MODEL" --resume || return $?
    step stage3 $PY scripts/03_regions.py --model "$MODEL"                  || return $?
    step stage7 $PY scripts/07_efficiency.py --model "$MODEL"               || return $?
}
chain_s4() { step stage4 $PY scripts/04_descriptive_accuracy.py --model "$MODEL" --resume; }
chain_s5() { step stage5 $PY scripts/05_sparsity.py --model "$MODEL" --resume; }
chain_s6() { step stage6 $PY scripts/06_stability.py --model "$MODEL" --resume; }
chain_final() {
    step stage9  $PY scripts/09_completeness.py --model "$MODEL"            || return $?
    step stage10 $PY scripts/10_explanation_correctness.py --model "$MODEL" || return $?
    step stage11_text  $PY scripts/11_report.py --model "$MODEL" --format text \
        --out "$XAI_RESULTS_DIR/report_$MODEL.txt"                         || return $?
    step stage11_latex $PY scripts/11_report.py --model "$MODEL" --format latex \
        --out "$XAI_RESULTS_DIR/report_$MODEL.tex"                         || return $?
}

chain_comp() {
    step comp_build  $PY scripts/01_build_sample.py --complement --seed 42 || return $?
    step comp_verify verify_complement                                      || return $?
    step comp_stage2 $PY scripts/02_baseline_inference.py --model "$MODEL" --resume \
        --manifest "$XAI_RESULTS_DIR/complement_manifest.csv" --out-tag complement || return $?
    step comp_report_text  $PY scripts/11_report.py --model "$MODEL" --format text \
        --out "$XAI_RESULTS_DIR/report_$MODEL.txt"                         || return $?
    step comp_report_latex $PY scripts/11_report.py --model "$MODEL" --format latex \
        --out "$XAI_RESULTS_DIR/report_$MODEL.tex"                         || return $?
}

submit() {
    sbatch --parsable --job-name="$1$SUFFIX" --export=ALL,XAI_CLUSTER_DIR="$HERE" \
        "$HERE/chain.sbatch" "$1"
}

# What a finished chain unlocks. mkdir is atomic, so when s4/s5/s6 finish at the
# same moment exactly one of them submits `final`.
after_chain() {
    case $1 in
        base) for c in s4 s5 s6; do echo "submitted $c: $(submit $c)"; done ;;
        s4|s5|s6)
            if [[ -e $STATE/s4.done && -e $STATE/s5.done && -e $STATE/s6.done ]] \
               && mkdir "$STATE/final.lock" 2>/dev/null; then
                echo "submitted final: $(submit final)"
            fi ;;
        comp) echo "== COMPLEMENT COMPLETE  $(date -Is)" ;;
        final) echo "== BATTERY COMPLETE  $(date -Is)" | tee "$STATE/COMPLETE" ;;
    esac
}
