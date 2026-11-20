"""Stage 7 -- Metric 4: Efficiency.

Median and p90 are the headline, not the mean. The mean is reported because it is
conventional; the p90 is what determines whether a deployment feels responsive,
and a long tail is invisible in an average.
"""

from __future__ import annotations

import statistics

NAN = float("nan")


def latency_stats(latencies_s: list[float]) -> dict:
    """Distribution summary. Empty input yields nan, never zero.

    Zero would read as "instantaneous"; nan reads as "not measured", which is what
    an empty set actually means.
    """
    if not latencies_s:
        return {"n": 0, "median": NAN, "p90": NAN, "mean": NAN, "max": NAN, "min": NAN}
    s = sorted(latencies_s)
    return {
        "n": len(s),
        "median": statistics.median(s),
        "p90": s[min(int(len(s) * 0.9), len(s) - 1)],
        "mean": statistics.mean(s),
        "max": s[-1],
        "min": s[0],
    }


def per_rule_stats(rows: list[dict]) -> dict:
    """Latency distribution per rule id."""
    by_rule: dict[str, list[float]] = {}
    for r in rows:
        raw = (r.get("latency_ms") or "").strip()
        if not raw:
            continue
        by_rule.setdefault(r["rule_id"], []).append(float(raw) / 1000.0)
    return {rule: latency_stats(v) for rule, v in sorted(by_rule.items())}
