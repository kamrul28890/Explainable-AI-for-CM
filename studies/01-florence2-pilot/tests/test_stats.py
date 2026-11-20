"""Unit tests for statistics hardening: bootstrap CIs, Wilcoxon, min-n (Phase 3.1)."""

import math

import numpy as np

from xai_pilot.stats import (
    bootstrap_ci,
    mean_pairwise_jaccard,
    meets_min_n,
    wilcoxon_signed_rank,
)


def test_bootstrap_ci_point_is_the_statistic():
    values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]  # mean 0.5
    result = bootstrap_ci(values, seed=0)
    assert abs(result["point"] - 0.5) < 1e-9
    assert result["n"] == 6


def test_bootstrap_ci_brackets_the_point():
    rng = np.random.default_rng(1)
    values = rng.normal(10.0, 2.0, size=200).tolist()
    result = bootstrap_ci(values, seed=0)
    assert result["lo"] < result["point"] < result["hi"]


def test_bootstrap_ci_is_reproducible_with_seed():
    values = list(np.random.default_rng(2).normal(0, 1, 100))
    a = bootstrap_ci(values, seed=42)
    b = bootstrap_ci(values, seed=42)
    assert a["lo"] == b["lo"] and a["hi"] == b["hi"]


def test_bootstrap_ci_constant_values_have_zero_width():
    result = bootstrap_ci([5.0, 5.0, 5.0], seed=0)
    assert result["lo"] == 5.0 and result["hi"] == 5.0


def test_bootstrap_ci_drops_nan():
    result = bootstrap_ci([1.0, float("nan"), 1.0], seed=0)
    assert result["n"] == 2
    assert abs(result["point"] - 1.0) < 1e-9


def test_bootstrap_ci_empty_is_nan():
    result = bootstrap_ci([], seed=0)
    assert result["n"] == 0
    assert math.isnan(result["point"])


def test_wilcoxon_detects_consistent_difference():
    a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = wilcoxon_signed_rank(a, b)
    assert result["p_value"] < 0.05
    assert result["n"] == 8


def test_wilcoxon_all_ties_is_not_significant():
    a = [1.0, 2.0, 3.0]
    result = wilcoxon_signed_rank(a, a)  # zero differences everywhere
    assert result["p_value"] == 1.0
    assert math.isnan(result["statistic"])


def test_meets_min_n_floor():
    assert meets_min_n(30) is True
    assert meets_min_n(29) is False
    assert meets_min_n(100, floor=50) is True


def test_mean_pairwise_jaccard_identical_sets_is_one():
    s = {"a", "b", "c"}
    assert mean_pairwise_jaccard([s, set(s), set(s)]) == 1.0


def test_mean_pairwise_jaccard_disjoint_sets_is_zero():
    assert mean_pairwise_jaccard([{"a", "b"}, {"c", "d"}]) == 0.0


def test_mean_pairwise_jaccard_half_overlap():
    # {a,b} vs {b,c}: intersection 1, union 3 -> 1/3.
    assert abs(mean_pairwise_jaccard([{"a", "b"}, {"b", "c"}]) - 1 / 3) < 1e-9


def test_mean_pairwise_jaccard_single_set_is_nan():
    assert math.isnan(mean_pairwise_jaccard([{"a"}]))
