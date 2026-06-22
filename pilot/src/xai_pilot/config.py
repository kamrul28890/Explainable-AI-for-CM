"""Shared paths, IDs, and constants for the XAI pilot."""

from pathlib import Path

PILOT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PILOT_ROOT / "data"
RESULTS_DIR = PILOT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_DIR = PILOT_ROOT / "report"

HF_DATASET_ID = "LouisChen15/ConstructionSite"
MODEL_ID = "microsoft/Florence-2-base-ft"

SEED = 42
SAMPLES_PER_CLASS = 50

# Priority order used by data.classify_image when a row violates more than
# one rule: the first matching class in this list wins.
RULE_TO_CLASS = {
    "rule_1_violation": "ppe_violation",
    "rule_2_violation": "fall_hazard",
    "rule_3_violation": "fall_hazard",
    "rule_4_violation": "struck_by_risk",
}
CLASS_PRIORITY = ["ppe_violation", "fall_hazard", "struck_by_risk", "compliant"]

for _dir in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, REPORT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
