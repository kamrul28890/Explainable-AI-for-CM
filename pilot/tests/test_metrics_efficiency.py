import math

from xai_pilot.metrics.efficiency import aggregate_timings, extrapolate


def test_aggregate_timings_basic_stats():
    stats = aggregate_timings([100.0, 200.0, 300.0])
    assert stats["n"] == 3
    assert stats["mean_ms"] == 200.0
    assert stats["median_ms"] == 200.0
    assert stats["max_ms"] == 300.0


def test_aggregate_timings_empty_list_is_nan():
    stats = aggregate_timings([])
    assert stats["n"] == 0
    assert math.isnan(stats["mean_ms"])


def test_extrapolate_scales_linearly():
    result = extrapolate(per_sample_ms=500.0, n_samples=1000)
    assert result["n_samples"] == 1000
    assert result["total_seconds"] == 500.0
    assert abs(result["total_minutes"] - 500.0 / 60) < 1e-9


def test_extrapolate_zero_samples_is_zero():
    result = extrapolate(per_sample_ms=500.0, n_samples=0)
    assert result["total_seconds"] == 0.0
