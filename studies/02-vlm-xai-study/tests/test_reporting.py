"""Publication-grade reporting: every rate carries an interval and an n."""
import pytest

from xai_vlm.metrics.reporting import (
    MIN_N_FLOOR,
    format_rate,
    proportion_ci,
    rate_row,
)


def test_proportion_ci_returns_point_lo_hi_and_n():
    d = proportion_ci([True] * 50 + [False] * 50, n_boot=500)
    assert d["n"] == 100
    assert d["point"] == pytest.approx(0.5, abs=0.01)
    assert d["lo"] < d["point"] < d["hi"]


def test_a_unanimous_sample_has_a_degenerate_interval():
    """All-successes is a real case (e.g. 100% parse rate). The bootstrap cannot
    manufacture uncertainty it has no evidence for, and must not crash."""
    d = proportion_ci([True] * 40, n_boot=500)
    assert d["point"] == 1.0
    assert d["lo"] == 1.0 and d["hi"] == 1.0


def test_empty_sample_is_nan_not_zero():
    d = proportion_ci([], n_boot=100)
    assert d["n"] == 0
    assert d["point"] != d["point"]  # nan


def test_wider_interval_for_smaller_n():
    """The whole point of reporting an interval: n=20 must look less certain
    than n=2000 at the same point estimate."""
    small = proportion_ci([True] * 10 + [False] * 10, n_boot=1000, seed=1)
    large = proportion_ci([True] * 1000 + [False] * 1000, n_boot=1000, seed=1)
    assert (small["hi"] - small["lo"]) > (large["hi"] - large["lo"])


def test_format_rate_shows_interval_and_n():
    s = format_rate(proportion_ci([True] * 59 + [False] * 41, n_boot=500))
    assert "%" in s and "[" in s and "n=100" in s


def test_rate_row_flags_below_floor_samples():
    """Below MIN_N_FLOOR a rate is not a stable estimate and must be marked, not
    silently reported as if it were."""
    row = rate_row("rule_4", [True] * 5 + [False] * 5, n_boot=200)
    assert row["below_floor"] is True
    assert "below reporting floor" in row["note"]


def test_rate_row_does_not_flag_adequate_samples():
    row = rate_row("rule_1", [True] * 60 + [False] * 60, n_boot=200)
    assert row["below_floor"] is False
    assert row["note"] == ""


def test_min_n_floor_matches_the_pilot():
    """The floor is shared with study 1 so the two studies report on the same
    standard."""
    from xai_pilot.stats import MIN_N_FLOOR as pilot_floor
    assert MIN_N_FLOOR == pilot_floor
