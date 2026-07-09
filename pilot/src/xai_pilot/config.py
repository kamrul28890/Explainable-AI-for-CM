"""Shared paths, IDs, and constants for the XAI pilot."""

from pathlib import Path

PILOT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PILOT_ROOT / "data"
# Local snapshot of the ConstructionSite dataset (parquet shards), used in
# preference to Hub streaming when present. See data.load_construction_site.
LOCAL_DATASET_DIR = DATA_DIR / "constructionsite"
RESULTS_DIR = PILOT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_DIR = PILOT_ROOT / "report"

HF_DATASET_ID = "LouisChen15/ConstructionSite"
MODEL_ID = "microsoft/Florence-2-base-ft"

SEED = 42
SAMPLES_PER_CLASS = 50

# Candidate-region ranking policy for masking/overlap metrics (Scale-up
# Phase 1.1). Default "area" reproduces the frozen 163-sample pilot record
# byte-for-byte; the scale run overrides this to "rule_aware" (via the metric
# scripts' --region-ranking flag) so masking tests the rule's queried object
# rather than the worker body. See regions.standardize_regions.
REGION_RANKING = "area"

# Class-labeling policy for sample selection (Scale-up Phase 1.2). Default
# "priority" reproduces the frozen pilot manifest (one class + one rule per
# image); "multilabel" counts an image toward every class it violates and
# tests it against every violated rule, recovering scarce classes (notably
# struck_by_risk) that priority-collapse hides. See data.select_balanced_sample.
LABELING = "priority"

# Compliant-image rule-assignment policy (Scale-up Phase 1.3). Default
# "round_robin" reproduces the frozen pilot (cycle all four rules blindly);
# "context_matched" routes excavator scenes to rule_4 and never assigns the
# vacuous rule_4 to excavator-free scenes, flagging metadata-unconfirmable
# assignments context_absent. See data.assign_compliant_rule.
COMPLIANT_ASSIGNMENT = "round_robin"

# Canonical decoding policy for the grounding path (Scale-up Phase 1.6). The
# pilot fragmented into three decoding regimes: baseline/masking/robustness used
# beam search (num_beams=3), attribution used greedy, stability used sampling.
# This unifies the deterministic path under one switch. Default "beam"
# reproduces the frozen pilot; the scale run uses "greedy" (the path attribution
# reads from, and shown to cost the same as beam for short grounding outputs).
# Stability's sampling and attribution's explicit greedy are unaffected (they
# pass num_beams explicitly). See model.run_task / _effective_num_beams.
DECODING = "beam"
DECODING_NUM_BEAMS = {"beam": 3, "greedy": 1}


def decoding_num_beams() -> int:
    """Number of beams implied by the current DECODING policy."""
    return DECODING_NUM_BEAMS[DECODING]


# Worker-loss-corrected metric reporting (Scale-up Phase 1.4). When True, the
# masking/perturbation metric scripts additionally log post-operation worker
# boxes and a flip_due_to_worker_loss flag, and report each metric twice: raw
# (all flips) and genuine (excluding flips caused by the worker becoming
# undetectable). Default False preserves the frozen pilot outputs; the scale
# run turns it on. See metrics.descriptive_accuracy / robustness / completeness.
REPORT_WORKER_LOSS_CORRECTED = False

# Priority order used by data.classify_image when a row violates more than
# one rule: the first matching class in this list wins.
RULE_TO_CLASS = {
    "rule_1_violation": "ppe_violation",
    "rule_2_violation": "fall_hazard",
    "rule_3_violation": "fall_hazard",
    "rule_4_violation": "struck_by_risk",
}
CLASS_PRIORITY = ["ppe_violation", "fall_hazard", "struck_by_risk", "compliant"]

# Create the standard artifact directories at import time so individual day
# scripts can write outputs without duplicating directory bootstrap logic.
for _dir in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, REPORT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
