"""Rebuilding a results tree in the original-image frame. No GPU, no model."""
import importlib.util
import sys
from pathlib import Path

import pytest

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "s17", STUDY_ROOT / "scripts" / "17_region_frame_fix.py")
s17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s17)

SIZE = (1200, 900)
FRAME = lambda size: (812, 588)  # noqa: E731


def test_baseline_region_is_re_parsed_from_the_reply():
    row = {"region": "(406.0, 294.0, 609.0, 441.0)", "verdict": "YES",
           "raw_text": "REASON: x\nREGION: (406, 294, 609, 441)\nANSWER: YES"}
    out, kind = s17.fix_baseline_row(row, SIZE, FRAME)
    assert out["region"] == "(600.0, 450.0, 900.0, 675.0)" and kind == "rescaled"
    assert out["verdict"] == "YES" and out["raw_text"] == row["raw_text"]


def test_baseline_without_size_is_left_alone():
    row = {"region": "(1.0, 2.0, 3.0, 4.0)", "raw_text": ""}
    out, kind = s17.fix_baseline_row(row, None, FRAME)
    assert out == row and kind == "no size"


def test_stability_regions_are_each_mapped():
    row = {"regions": "(406.0, 294.0, 609.0, 441.0)||(0.0, 0.0, 406.0, 294.0)"}
    out = s17.fix_stability_row(row, SIZE, FRAME)
    assert out["regions"] == "(600.0, 450.0, 900.0, 675.0)||(0.0, 0.0, 600.0, 450.0)"


def test_robustness_drift_is_recomputed_in_the_original_frame():
    row = {"baseline_region": "(0.0, 0.0, 203.0, 147.0)", "region_blur": "(203.0, 147.0, 406.0, 294.0)",
           "survived_blur": "True", "drift_blur": "0.0000", "drifted_blur": "False"}
    out = s17.fix_robustness_row(row, SIZE, FRAME)
    # centroids (150, 112.5) and (450, 337.5): 375 px over a 1500 px diagonal
    assert float(out["drift_blur"]) == pytest.approx(0.25, abs=1e-4)
    assert out["drifted_blur"] == "True"
