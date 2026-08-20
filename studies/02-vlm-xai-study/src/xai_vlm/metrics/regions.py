"""Stage 3 -- what the model pointed at, and how big the image was.

Two jobs. The obvious one is classifying whether a region was offered at all; the
NO-REGION RATE is a headline result in its own right, because a model with high
accuracy and no explanations is not explainable whatever its accuracy.

The less obvious one is recording image dimensions. Metric 7 compares a normalised
annotation box against an absolute-pixel model region, and neither can be
converted without width and height -- which Stage 2 does not store.
"""

from __future__ import annotations

REGION_FIELDS = [
    "image_id", "rule_id", "verdict", "source", "region", "image_w", "image_h",
]


def classify_source(region: str) -> str:
    """`model` when a box was returned, `none` when it was not."""
    return "model" if (region or "").strip() else "none"


def no_region_rate(rows: list[dict]) -> float:
    """Share of rows where no explanation region was offered."""
    if not rows:
        return float("nan")
    return sum(1 for r in rows if r["source"] == "none") / len(rows)


def build_region_rows(baseline_rows: list[dict],
                      sizes: dict[str, tuple[int, int]]) -> list[dict]:
    """One region row per baseline row.

    Unknown dimensions are written as 0 rather than a plausible default. A guessed
    1200x900 would silently produce wrong normalisation for every portrait image;
    a zero is visible and forces the row to be excluded downstream.
    """
    out = []
    for r in baseline_rows:
        w, h = sizes.get(r["image_id"], (0, 0))
        out.append({
            "image_id": r["image_id"],
            "rule_id": r["rule_id"],
            "verdict": r.get("verdict", ""),
            "source": classify_source(r.get("region", "")),
            "region": r.get("region", ""),
            "image_w": w,
            "image_h": h,
        })
    return out
