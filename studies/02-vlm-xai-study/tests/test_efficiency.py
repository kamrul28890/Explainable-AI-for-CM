"""Stage 7: latency distribution. Pure arithmetic."""
import pytest

from xai_vlm.metrics.efficiency import latency_stats, per_rule_stats


def test_latency_stats_reports_median_and_p90_not_just_mean():
    """The tail decides whether a deployment feels responsive."""
    s = latency_stats([1.0] * 90 + [100.0] * 10)
    assert s["n"] == 100
    assert s["median"] == pytest.approx(1.0)
    assert s["p90"] == pytest.approx(100.0)
    assert s["mean"] == pytest.approx(10.9)
    assert s["max"] == pytest.approx(100.0)


def test_latency_stats_empty_is_nan_not_zero():
    """Zero would read as instantaneous; nan reads as absent."""
    s = latency_stats([])
    assert s["n"] == 0
    assert s["median"] != s["median"]  # nan


def test_per_rule_stats_splits_by_rule():
    rows = [
        {"rule_id": "rule_1", "latency_ms": "1000"},
        {"rule_id": "rule_1", "latency_ms": "3000"},
        {"rule_id": "rule_4", "latency_ms": "5000"},
    ]
    out = per_rule_stats(rows)
    assert out["rule_1"]["n"] == 2
    assert out["rule_1"]["median"] == pytest.approx(2.0)
    assert out["rule_4"]["n"] == 1


def test_per_rule_stats_ignores_rows_with_no_latency():
    rows = [{"rule_id": "rule_1", "latency_ms": ""},
            {"rule_id": "rule_1", "latency_ms": "2000"}]
    assert per_rule_stats(rows)["rule_1"]["n"] == 1


# --- the caveat printed with every efficiency figure -------------------------
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stage7", Path(__file__).resolve().parents[1] / "scripts" / "07_efficiency.py")
stage7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage7)


def test_run_device_comes_from_the_baseline_fingerprint(tmp_path):
    (tmp_path / "baseline_m.state.json").write_text(json.dumps({"device": "NVIDIA A40"}))
    assert stage7.run_device(tmp_path, "m") == "NVIDIA A40"


def test_frozen_run_without_a_device_key_is_the_local_3070(tmp_path):
    """State files written before 2026-11-09 carry no device; all are local."""
    (tmp_path / "baseline_m.state.json").write_text(json.dumps({"seed": 42}))
    assert stage7.run_device(tmp_path, "m") == stage7.STANDALONE_DEVICE


def test_caveat_quotes_the_ratio_only_on_the_machine_that_measured_standalone():
    same = "\n".join(stage7.latency_caveat(10.1, stage7.STANDALONE_DEVICE))
    assert "3.3x" in same

    other = "\n".join(stage7.latency_caveat(2.3, "NVIDIA A40"))
    assert "x longer" not in other           # no cross-GPU ratio
    assert "NVIDIA A40" in other and stage7.STANDALONE_DEVICE in other


def test_caveat_no_longer_calls_the_gap_unexplained_machine_independent():
    """The A40 run (2.3 s in-run) showed the gap belongs to the local machine."""
    text = "\n".join(stage7.latency_caveat(10.1, stage7.STANDALONE_DEVICE))
    assert "scholar-a40" in text
