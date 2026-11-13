"""Statistics hardening for headline metrics (Scale-up Phase 3).

The plan requires every headline number to carry a bootstrap confidence interval
and, for paired comparisons, a nonparametric significance test (Wilcoxon
signed-rank, matching Prof. Abdallah's prior IDS work), and to be reported only
when a minimum samples-per-class floor is met. These helpers provide those three
pieces; they are pure and unit-tested so a metric's CI/p-value is reproducible.
"""

import itertools
from collections.abc import Callable

import numpy as np
from scipy.stats import wilcoxon as _scipy_wilcoxon

# Recommended minimum samples-per-class floor before a rate is reported as a
# stable estimate (plan Phase 0 statistics standard).
MIN_N_FLOOR = 30


def bootstrap_ci(
    values,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap confidence interval for a statistic of `values`.

    NaNs are dropped. Returns {point, lo, hi, n, ci}. Reproducible for a fixed
    seed. A constant sample yields a zero-width interval.
    """
    arr = np.asarray([v for v in values if v == v], dtype=float)  # drop NaN
    if arr.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "ci": ci}
    point = float(statistic(arr))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = statistic(rng.choice(arr, size=arr.size, replace=True))
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n": int(arr.size), "ci": ci}


def wilcoxon_signed_rank(a, b) -> dict:
    """Wilcoxon signed-rank test on paired samples `a` vs `b`.

    Returns {statistic, p_value, n}. When every paired difference is zero (no
    signal), the test is undefined, so this returns p_value 1.0 and a NaN
    statistic rather than raising.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    n = int(diff.size)
    if n == 0 or np.all(diff == 0):
        return {"statistic": float("nan"), "p_value": 1.0, "n": n}
    stat, p = _scipy_wilcoxon(a, b)
    return {"statistic": float(stat), "p_value": float(p), "n": n}


def meets_min_n(n: int, floor: int = MIN_N_FLOOR) -> bool:
    """True if a class/subgroup has enough samples to report as a stable estimate."""
    return n >= floor


def mean_pairwise_jaccard(sets: list[set]) -> float:
    """Mean pairwise Jaccard overlap across a list of sets (Scale-up Phase 3.2).

    Used to quantify how stable a seeded sample selection is across seeds: 1.0
    means every seed picks the same samples, 0.0 means disjoint selections. NaN
    for fewer than two sets.
    """
    if len(sets) < 2:
        return float("nan")
    scores = []
    for a, b in itertools.combinations(sets, 2):
        union = a | b
        scores.append(len(a & b) / len(union) if union else 1.0)
    return sum(scores) / len(scores)
