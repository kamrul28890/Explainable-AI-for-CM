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
