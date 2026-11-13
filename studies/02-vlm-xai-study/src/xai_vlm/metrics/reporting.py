"""Publication-grade reporting: no bare point estimates.

Every rate in this study is an estimate from a finite sample, and a paper that
prints "59%" without an interval invites the reader to treat it as exact. Worse,
it hides the difference between a rate measured on 2,555 rows and one measured on
24 -- which matters here, because rule_2 and rule_4 have an order of magnitude
fewer positives than rule_1.

The bootstrap and the minimum-n floor are imported from the pilot's `stats`
module rather than reimplemented, so both studies report on the same standard and
a reviewer comparing them is comparing like with like.
"""

from __future__ import annotations

from xai_pilot.stats import MIN_N_FLOOR, bootstrap_ci

__all__ = ["MIN_N_FLOOR", "proportion_ci", "format_rate", "rate_row"]


def proportion_ci(flags, n_boot: int = 10_000, ci: float = 0.95,
                  seed: int = 0) -> dict:
    """Percentile bootstrap interval for a proportion.

    `flags` is any iterable of truthy/falsey values -- one per observation.

    A unanimous sample yields a zero-width interval. That is honest rather than
    broken: with 40 successes out of 40 the data contains no evidence of
    variation, and manufacturing an interval would invent uncertainty the sample
    does not show. The `n` travels with the estimate so a reader can judge it.
    """
    values = [1.0 if f else 0.0 for f in flags]
    return bootstrap_ci(values, n_boot=n_boot, ci=ci, seed=seed)


def format_rate(d: dict, digits: int = 1) -> str:
    """'59.0% [57.1, 60.9] (n=2555)' -- the form every headline should take."""
    if not d["n"]:
        return "n/a (n=0)"
    return (f"{d['point'] * 100:.{digits}f}% "
            f"[{d['lo'] * 100:.{digits}f}, {d['hi'] * 100:.{digits}f}] "
            f"(n={d['n']})")


def rate_row(label: str, flags, n_boot: int = 10_000, seed: int = 0) -> dict:
    """One reportable rate, with its interval and a below-floor warning.

    Below MIN_N_FLOOR the estimate is not stable enough to report as a finding.
    It is still computed and shown -- suppressing it would hide a real subgroup --
    but it carries a visible note so it cannot be quoted as though it were solid.
    """
    d = proportion_ci(flags, n_boot=n_boot, seed=seed)
    below = 0 < d["n"] < MIN_N_FLOOR
    return {
        "label": label,
        "point": d["point"],
        "lo": d["lo"],
        "hi": d["hi"],
        "n": d["n"],
        "below_floor": below,
        "note": (f"below reporting floor of {MIN_N_FLOOR}; illustrative only"
                 if below else ""),
        "formatted": format_rate(d),
    }
