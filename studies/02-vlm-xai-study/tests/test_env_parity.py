"""scripts/21_env_parity.py: agreement between two environments' Stage 2 rows."""
import importlib.util

import pytest

from xai_vlm.config import STUDY_ROOT

spec = importlib.util.spec_from_file_location("envp", STUDY_ROOT / "scripts" / "21_env_parity.py")
envp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(envp)


def _row(i, raw="r", verdict="YES", region="(0, 0, 10, 10)", expected="YES"):
    return {"image_id": f"{i:07d}", "rule_id": "rule_1", "raw_text": raw, "verdict": verdict,
            "region": region, "expected": expected}


def test_identical_runs_agree_everywhere():
    rows = [_row(i) for i in range(3)]
    r = envp.compare(rows, [dict(x) for x in rows])
    assert (r["rows"], r["raw_identical"], r["verdict_agree"], r["region_both"]) == (3, 3, 3, 3)
    assert r["region_iou"] == [1.0, 1.0, 1.0]


def test_differences_are_counted_by_kind():
    a = [_row(0), _row(1), _row(2, region=""), _row(3)]
    b = [_row(0, raw="other"), _row(1, verdict="NO", region=""), _row(2, region=""),
         _row(4)]
    r = envp.compare(a, b)
    assert (r["rows"], r["only_a"], r["only_b"]) == (3, 1, 1)
    assert r["raw_identical"] == 2 and r["verdict_agree"] == 2
    assert (r["region_both"], r["region_one"], r["region_neither"]) == (1, 1, 1)
    assert r["correct_a"] == 3 and r["correct_b"] == 2


def test_iou_of_half_overlapping_boxes():
    assert envp.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
