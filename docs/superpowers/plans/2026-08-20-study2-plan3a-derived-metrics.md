# Study 2 Plan 3a — Derived Metrics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute Stages 3, 7 and 10 from the completed baseline, with no further model inference.

**Architecture:** Four small modules under `src/xai_vlm/metrics/`, each with one responsibility, driven by three thin stage scripts. The embedding model is isolated behind a tiny interface so the verdict logic is tested on exact arithmetic rather than on what MiniLM happens to output.

**Tech Stack:** Python 3.10, pytest, sentence-transformers 5.7.0 (CPU), PIL, numpy.

**Spec:** `docs/superpowers/specs/2026-08-20-study2-plan3a-derived-metrics-design.md`

## Global Constraints

- **Work from `studies/02-vlm-xai-study/`.** Python is `../../pilot/.venv/Scripts/python`.
- **No GPU.** Nothing in this plan imports `xai_vlm.backends.qwen` or loads a VLM.
- **Do not modify** `rules.py`, `parsing.py`, `sampling.py`, `masking.py`, `runner.py`, `backends/`. If one seems wrong, stop and report.
- **Do not re-run Stage 2** or modify `results/baseline_qwen7b-4bit.csv` (committed, sha in its `.state.json`).
- **Pre-registered thresholds, not tunable:** `SIMILARITY_FLOOR = 0.60`, `IOU_FLOOR = 0.30`. Never adjust to fit observed results.
- **Coordinate convention:** annotation boxes are normalised 0–1; model regions are absolute pixels. Convert model → normalised using image dimensions. Getting this wrong puts every box in the top-left corner.
- **Undefined ≠ incorrect.** Rows the metric cannot cover are counted separately, never coerced.
- **TDD.** Failing test, watch it fail, implement, watch it pass, commit.
- Commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `src/xai_vlm/metrics/regions.py` | Classify region source; no-region rate. |
| `src/xai_vlm/metrics/efficiency.py` | Latency distribution statistics. |
| `src/xai_vlm/metrics/geometry.py` | Box coordinate conversion and IoU. |
| `src/xai_vlm/metrics/text_similarity.py` | Embedding interface + pinned MiniLM implementation. |
| `src/xai_vlm/metrics/correctness.py` | Three signals → conjunctive verdict. |
| `scripts/03_regions.py` | Stage 3 runner (adds image dimensions). |
| `scripts/07_efficiency.py` | Stage 7 runner. |
| `scripts/10_explanation_correctness.py` | Stage 10 runner. |

---

### Task 1: Geometry — coordinate conversion and IoU

**Files:**
- Create: `src/xai_vlm/metrics/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Produces:
  - `def to_normalised(box: tuple[float,float,float,float], width: int, height: int) -> tuple[float,float,float,float]`
  - `def iou(a, b) -> float` — both boxes in the same coordinate space
  - `def max_iou(model_box_norm, human_boxes_norm: list) -> float` — 0.0 when the list is empty

- [ ] **Step 1: Write the failing test**

```python
# studies/02-vlm-xai-study/tests/test_geometry.py
"""Box arithmetic for metric 7. Pure functions, no model."""
import pytest

from xai_vlm.metrics.geometry import iou, max_iou, to_normalised


def test_to_normalised_divides_by_image_size():
    assert to_normalised((600, 450, 1200, 900), 1200, 900) == (0.5, 0.5, 1.0, 1.0)


def test_normalising_is_not_the_identity():
    """The top-left-corner bug: treating pixels as fractions puts every box at 0,0.

    A pixel box (600, 450, 720, 540) on a 1200x900 image is the middle of the
    frame. Read as normalised it would be far outside 0-1. If these ever compare
    equal, the conversion was skipped.
    """
    pixel_box = (600, 450, 720, 540)
    assert to_normalised(pixel_box, 1200, 900) != pixel_box


def test_iou_identical_boxes_is_one():
    assert iou((0.1, 0.1, 0.5, 0.5), (0.1, 0.1, 0.5, 0.5)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou((0.0, 0.0, 0.1, 0.1), (0.8, 0.8, 0.9, 0.9)) == 0.0


def test_iou_half_overlap():
    # a covers x 0..0.2, b covers x 0.1..0.3, same y. Intersection 0.1 wide,
    # union 0.3 wide -> 1/3.
    assert iou((0.0, 0.0, 0.2, 0.1), (0.1, 0.0, 0.3, 0.1)) == pytest.approx(1 / 3)


def test_max_iou_takes_the_best_annotated_box():
    """A photo with three violations: matching any one of them is success."""
    model = (0.5, 0.5, 0.6, 0.6)
    humans = [(0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6), (0.9, 0.9, 1.0, 1.0)]
    assert max_iou(model, humans) == pytest.approx(1.0)


def test_max_iou_with_no_annotated_boxes_is_zero():
    assert max_iou((0.1, 0.1, 0.2, 0.2), []) == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xai_vlm.metrics.geometry'`

- [ ] **Step 3: Write the implementation**

```python
# studies/02-vlm-xai-study/src/xai_vlm/metrics/geometry.py
"""Box arithmetic for metric 7.

THE COORDINATE TRAP. Dataset annotation boxes are normalised to 0-1 as fractions
of width and height; the model emits absolute pixels. Comparing them without
converting puts every annotation in the top-left corner and silently produces an
IoU near zero for every row -- a study-wide null result that looks like a finding.
The architecture document calls this out specifically (section 5.2).

Everything here works in NORMALISED space. Convert once, at the edge.
"""

from __future__ import annotations

Box = tuple[float, float, float, float]


def to_normalised(box: Box, width: int, height: int) -> Box:
    """Absolute-pixel box -> fraction-of-image box."""
    x0, y0, x1, y1 = box
    return (x0 / width, y0 / height, x1 / width, y1 / height)


def iou(a: Box, b: Box) -> float:
    """Intersection over union. Both boxes must be in the same space."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def max_iou(model_box: Box, human_boxes: list) -> float:
    """Best agreement with any annotated box.

    The model is asked for the single most relevant area, so agreeing with any one
    annotated instance is success. Averaging over boxes would penalise a correct
    answer on a photograph containing three separate violations.
    """
    if not human_boxes:
        return 0.0
    return max(iou(model_box, tuple(h)) for h in human_boxes)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_geometry.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add studies/02-vlm-xai-study/src/xai_vlm/metrics/geometry.py studies/02-vlm-xai-study/tests/test_geometry.py
git commit -m "feat(study2): add box geometry for metric 7

Annotation boxes are normalised 0-1, model regions are absolute pixels.
Converting is one-way at the edge, and a test asserts the conversion is not the
identity -- skipping it would put every box in the top-left corner and produce a
study-wide null result that looks like a finding.

IoU against multiple annotated boxes takes the maximum: the model is asked for
one region, and matching any annotated instance is success.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Stage 3 — regions and image dimensions

**Files:**
- Create: `src/xai_vlm/metrics/regions.py`, `scripts/03_regions.py`
- Test: `tests/test_regions.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `REGION_FIELDS: list[str]`
  - `def classify_source(region: str) -> str` — `"model"` or `"none"`
  - `def no_region_rate(rows: list[dict]) -> float`
  - `def build_region_rows(baseline_rows: list[dict], sizes: dict[str, tuple[int,int]]) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# studies/02-vlm-xai-study/tests/test_regions.py
"""Stage 3: region source and image dimensions. No model, no image decode."""
from xai_vlm.metrics.regions import (
    REGION_FIELDS,
    build_region_rows,
    classify_source,
    no_region_rate,
)


def test_classify_source_model_when_a_box_is_present():
    assert classify_source("(10.0, 20.0, 30.0, 40.0)") == "model"


def test_classify_source_none_when_empty():
    assert classify_source("") == "none"
    assert classify_source("   ") == "none"


def test_no_region_rate():
    rows = [{"source": "model"}, {"source": "none"}, {"source": "model"}, {"source": "none"}]
    assert no_region_rate(rows) == 0.5


def test_build_region_rows_carries_dimensions_and_source():
    baseline = [
        {"image_id": "0001", "rule_id": "rule_1", "verdict": "YES",
         "region": "(10.0, 20.0, 30.0, 40.0)"},
        {"image_id": "0002", "rule_id": "rule_3", "verdict": "NO", "region": ""},
    ]
    sizes = {"0001": (1200, 900), "0002": (675, 1200)}
    rows = build_region_rows(baseline, sizes)

    assert [r["source"] for r in rows] == ["model", "none"]
    assert rows[0]["image_w"] == 1200 and rows[0]["image_h"] == 900
    assert rows[1]["image_w"] == 675 and rows[1]["image_h"] == 1200
    assert list(rows[0]) == REGION_FIELDS


def test_missing_dimensions_are_recorded_as_zero_not_guessed():
    """An image the size pass could not read must be visible, not defaulted."""
    baseline = [{"image_id": "9999", "rule_id": "rule_1", "verdict": "YES",
                 "region": "(1.0, 2.0, 3.0, 4.0)"}]
    rows = build_region_rows(baseline, {})
    assert rows[0]["image_w"] == 0 and rows[0]["image_h"] == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_regions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xai_vlm.metrics.regions'`

- [ ] **Step 3: Write the implementation**

```python
# studies/02-vlm-xai-study/src/xai_vlm/metrics/regions.py
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_regions.py -v`
Expected: 5 passed

- [ ] **Step 5: Write the stage script**

```python
# studies/02-vlm-xai-study/scripts/03_regions.py
"""Stage 3 -- explanation regions, and the image dimensions metric 7 needs.

    python scripts/03_regions.py --model qwen7b-4bit
"""
import argparse
import csv
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402
from xai_vlm.metrics.regions import REGION_FIELDS, build_region_rows, no_region_rate  # noqa: E402


def read_image_sizes(loader, wanted: set) -> dict:
    """image_id -> (width, height), read from headers rather than full decodes.

    PIL opens lazily: `.size` is available from the header without decoding pixel
    data, so this is I/O-bound and far cheaper than decoding 2,556 photographs.
    """
    sizes = {}
    for split in config.SPLITS:
        for rec in loader(split=split, streaming=True):
            iid = rec["image_id"]
            if iid in wanted and iid not in sizes:
                sizes[iid] = rec["image"].size
            if len(sizes) >= len(wanted):
                return sizes
    return sizes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen7b-4bit")
    args = ap.parse_args()

    baseline_path = STUDY_ROOT / "results" / f"baseline_{args.model}.csv"
    if not baseline_path.exists():
        print(f"no baseline at {baseline_path}. Run Stage 2 first.", file=sys.stderr)
        return 1

    with baseline_path.open(newline="", encoding="utf-8") as fh:
        baseline = list(csv.DictReader(fh))
    print(f"baseline: {len(baseline)} rows")

    from xai_pilot.data import load_construction_site

    wanted = {r["image_id"] for r in baseline}
    print(f"reading dimensions for {len(wanted)} images (headers only) ...")
    sizes = read_image_sizes(load_construction_site, wanted)
    print(f"  got {len(sizes)}")

    rows = build_region_rows(baseline, sizes)
    out = STUDY_ROOT / "results" / f"regions_{args.model}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REGION_FIELDS)
        w.writeheader()
        w.writerows(rows)

    missing_dims = sum(1 for r in rows if not r["image_w"])
    print("\n" + "=" * 60)
    print(f"NO-REGION RATE : {no_region_rate(rows):.1%}")
    print("  the share of cases where the system offered no explanation at all")
    yes = [r for r in rows if r["verdict"] == "YES"]
    if yes:
        print(f"  among YES answers : {no_region_rate(yes):.1%}")
    if missing_dims:
        print(f"WARNING: {missing_dims} rows have no image dimensions; "
              "metric 7 will exclude them.")
    print(f"wrote {out}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run Stage 3 for real**

Run: `../../pilot/.venv/Scripts/python scripts/03_regions.py --model qwen7b-4bit`
Expected: `results/regions_qwen7b-4bit.csv` with 3,882 rows, no-region rate ≈ 21.9%, and **0 rows with missing dimensions**. If any dimensions are missing, stop and report rather than proceeding.

- [ ] **Step 7: Commit**

```bash
git add studies/02-vlm-xai-study/src/xai_vlm/metrics/regions.py studies/02-vlm-xai-study/tests/test_regions.py studies/02-vlm-xai-study/scripts/03_regions.py studies/02-vlm-xai-study/results/regions_qwen7b-4bit.csv
git commit -m "feat(study2): Stage 3 -- explanation regions and image dimensions

The no-region rate is a headline in itself: the share of cases where the system
could not explain itself at all.

Also records image dimensions, which Stage 2 never stored and metric 7 cannot
convert coordinates without. Read from image headers rather than full decodes.
Unknown dimensions are written as 0, never a plausible default -- a guessed
1200x900 would silently mis-normalise every portrait image.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Stage 7 — efficiency

**Files:**
- Create: `src/xai_vlm/metrics/efficiency.py`, `scripts/07_efficiency.py`
- Test: `tests/test_efficiency.py`

**Interfaces:**
- Produces:
  - `def latency_stats(latencies_s: list[float]) -> dict` — keys `n, median, p90, mean, max, min`
  - `def per_rule_stats(rows: list[dict]) -> dict[str, dict]`

- [ ] **Step 1: Write the failing test**

```python
# studies/02-vlm-xai-study/tests/test_efficiency.py
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_efficiency.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xai_vlm.metrics.efficiency'`

- [ ] **Step 3: Write the implementation**

```python
# studies/02-vlm-xai-study/src/xai_vlm/metrics/efficiency.py
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_efficiency.py -v`
Expected: 4 passed

- [ ] **Step 5: Write the stage script**

```python
# studies/02-vlm-xai-study/scripts/07_efficiency.py
"""Stage 7 -- Metric 4: Efficiency, from timings already recorded.

    python scripts/07_efficiency.py --model qwen7b-4bit
"""
import argparse
import csv
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm.metrics.efficiency import latency_stats, per_rule_stats  # noqa: E402

# Measured standalone on the same images, for the caveat below. See the Plan 3a
# design doc and runner.py: identical work costs 3.1 s alone and ~10 s in a run.
STANDALONE_MEDIAN_S = 3.1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen7b-4bit")
    args = ap.parse_args()

    path = STUDY_ROOT / "results" / f"baseline_{args.model}.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    lat = [float(r["latency_ms"]) / 1000.0 for r in rows if (r.get("latency_ms") or "").strip()]
    overall = latency_stats(lat)
    by_rule = per_rule_stats(rows)

    out = STUDY_ROOT / "results" / f"efficiency_{args.model}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scope", "n", "median_s", "p90_s",
                                           "mean_s", "min_s", "max_s"])
        w.writeheader()
        w.writerow({"scope": "overall", "n": overall["n"],
                    "median_s": round(overall["median"], 2),
                    "p90_s": round(overall["p90"], 2),
                    "mean_s": round(overall["mean"], 2),
                    "min_s": round(overall["min"], 2),
                    "max_s": round(overall["max"], 2)})
        for rule, s in by_rule.items():
            w.writerow({"scope": rule, "n": s["n"],
                        "median_s": round(s["median"], 2),
                        "p90_s": round(s["p90"], 2),
                        "mean_s": round(s["mean"], 2),
                        "min_s": round(s["min"], 2),
                        "max_s": round(s["max"], 2)})

    print("=" * 66)
    print(f"median {overall['median']:.1f}s   p90 {overall['p90']:.1f}s   "
          f"mean {overall['mean']:.1f}s   max {overall['max']:.1f}s   n={overall['n']}")
    for rule, s in by_rule.items():
        print(f"  {rule}: median {s['median']:.1f}s  p90 {s['p90']:.1f}s  n={s['n']}")
    print()
    print("CAVEAT, reported with every efficiency figure:")
    print(f"  in-run median      {overall['median']:.1f}s   <- PRIMARY, what a batch costs")
    print(f"  standalone median  {STANDALONE_MEDIAN_S:.1f}s   <- same images, measured alone")
    print("  The 3.5x gap is real, reproducible and unexplained. Ruled out: VRAM")
    print("  (6.97 GB peak of 8, stable), system RAM (18 GB free), image size, and")
    print("  reply length. Quoting only the standalone figure understates deployment")
    print("  cost; quoting only the in-run figure charges the model for harness")
    print("  overhead. Report both.")
    print(f"wrote {out}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run Stage 7 for real**

Run: `../../pilot/.venv/Scripts/python scripts/07_efficiency.py --model qwen7b-4bit`
Expected: overall median ≈ 10.1 s, p90 ≈ 26.5 s, max ≈ 477 s, n = 3,882.

- [ ] **Step 7: Commit**

```bash
git add studies/02-vlm-xai-study/src/xai_vlm/metrics/efficiency.py studies/02-vlm-xai-study/tests/test_efficiency.py studies/02-vlm-xai-study/scripts/07_efficiency.py studies/02-vlm-xai-study/results/efficiency_qwen7b-4bit.csv
git commit -m "feat(study2): Stage 7 -- efficiency, median and p90 over mean

The tail decides whether a deployment feels responsive, so p90 is reported
alongside the median and the mean is included only because it is conventional.

Both the in-run and standalone medians are printed together. Identical work costs
3.1s alone and ~10s inside a run; quoting either alone misrepresents the cost in
opposite directions. The gap is documented as unexplained rather than smoothed.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Text similarity behind an interface

**Files:**
- Create: `src/xai_vlm/metrics/text_similarity.py`
- Modify: `requirements/models.lock`
- Test: `tests/test_text_similarity.py`

**Interfaces:**
- Produces:
  - `EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"`
  - `class Embedder(Protocol)` with `def encode(self, texts: list[str]) -> np.ndarray`
  - `class FakeEmbedder` — maps exact strings to vectors, for tests
  - `class MiniLMEmbedder` — the real one, loaded lazily with a pinned revision
  - `def cosine(a: np.ndarray, b: np.ndarray) -> float`

- [ ] **Step 1: Write the failing test**

```python
# studies/02-vlm-xai-study/tests/test_text_similarity.py
"""The embedding boundary. The real model is exercised in one marked test only."""
import numpy as np
import pytest

from xai_vlm.metrics.text_similarity import FakeEmbedder, cosine


def test_cosine_of_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_of_zero_vector_is_zero_not_nan():
    """An empty rationale must score 0, not poison the arithmetic with nan."""
    assert cosine(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0


def test_fake_embedder_returns_scripted_vectors():
    fake = FakeEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    out = fake.encode(["a", "b"])
    assert out.shape == (2, 2)
    assert cosine(out[0], out[1]) == pytest.approx(0.0)


def test_fake_embedder_unknown_text_is_orthogonal_to_everything():
    """Unscripted text must not accidentally score as similar."""
    fake = FakeEmbedder({"a": [1.0, 0.0, 0.0]})
    out = fake.encode(["a", "never seen"])
    assert cosine(out[0], out[1]) == pytest.approx(0.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_text_similarity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xai_vlm.metrics.text_similarity'`

- [ ] **Step 3: Write the implementation**

```python
# studies/02-vlm-xai-study/src/xai_vlm/metrics/text_similarity.py
"""Sentence embeddings for metric 7, behind an interface.

The interface exists so the VERDICT LOGIC can be tested on exact arithmetic. If
correctness.py imported a real model, its tests would assert on whatever MiniLM
happens to output for a given sentence -- which is neither stable across versions
nor a statement about the code under test.

Lexical overlap is not an option here. The architecture document's own worked
example is two sentences meant to score 0.88 that share almost no vocabulary
("not wearing a hard hat" vs "has no helmet on"). TF-IDF scores that near zero and
would mark a correct explanation incorrect.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

# Pinned in requirements/models.lock, like every other weight in this study. An
# unpinned embedding model silently changes metric 7 between runs.
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """One row of floats per input string."""
        ...


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, with an empty vector scoring 0 rather than nan.

    A missing or empty rationale is a real case in this data. Returning nan would
    propagate into every aggregate; returning 0 says "no similarity", which is the
    truthful reading.
    """
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class FakeEmbedder:
    """Scripted embeddings for tests.

    Unknown text gets a unique basis vector, so anything unscripted is orthogonal
    to everything else. A fake that returned zeros would make unscripted text
    score 0 against a real vector too, hiding the difference between "dissimilar"
    and "not configured".
    """

    def __init__(self, table: dict[str, list[float]], dim: int = 8):
        self._table = {k: np.asarray(v, dtype=float) for k, v in table.items()}
        self._dim = max(dim, max((len(v) for v in self._table.values()), default=dim))
        self._unknown: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            if t in self._table:
                v = np.zeros(self._dim)
                src = self._table[t]
                v[:len(src)] = src
            else:
                if t not in self._unknown:
                    idx = self._dim - 1 - (len(self._unknown) % self._dim)
                    v = np.zeros(self._dim)
                    v[idx] = 1.0
                    self._unknown[t] = v
                v = self._unknown[t]
            out.append(v)
        return np.vstack(out)


class MiniLMEmbedder:
    """The real embedder. Loaded lazily so importing this module costs nothing."""

    def __init__(self, model_id: str = EMBED_MODEL_ID, revision: str | None = None):
        self._model_id = model_id
        self._revision = revision
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            kw = {"revision": self._revision} if self._revision else {}
            self._model = SentenceTransformer(self._model_id, device="cpu", **kw)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._load().encode(texts, show_progress_bar=False))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_text_similarity.py -v`
Expected: 5 passed

- [ ] **Step 5: Download the embedding model and record its revision**

```bash
../../pilot/.venv/Scripts/python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('sentence-transformers/all-MiniLM-L6-v2')
print(p)
"
```

Then read the resolved commit sha from the snapshot path (the final path component under `snapshots/`) and append to `requirements/models.lock`:

```
# Metric 7 sentence embeddings. Unpinned, this silently changes explanation
# correctness scores between runs.
minilm-l6-v2  sentence-transformers/all-MiniLM-L6-v2  <sha>  # verified <date>
```

- [ ] **Step 6: Verify the real model behaves as the design assumes**

```bash
../../pilot/.venv/Scripts/python -c "
import sys; sys.path.insert(0,'src')
from xai_vlm.metrics.text_similarity import MiniLMEmbedder, cosine
e = MiniLMEmbedder()
v = e.encode([
  'The mechanic in front of the truck is not wearing a hard hat.',
  'The man working beside the truck cab has no helmet on.',
  'The excavation edge on the right is not guarded.',
])
print('paraphrase pair :', round(cosine(v[0], v[1]), 3), ' (design expects > 0.60)')
print('unrelated pair  :', round(cosine(v[0], v[2]), 3), ' (design expects < 0.60)')
"
```

Expected: the paraphrase pair scores above 0.60 and the unrelated pair below it. **If the paraphrase pair scores below 0.60, stop and report** — the pre-registered threshold would be mis-calibrated for this model, and that is a decision for the user, not a number to quietly adjust.

- [ ] **Step 7: Commit**

```bash
git add studies/02-vlm-xai-study/src/xai_vlm/metrics/text_similarity.py studies/02-vlm-xai-study/tests/test_text_similarity.py requirements/models.lock
git commit -m "feat(study2): add pinned sentence embeddings behind an interface

The interface exists so the metric 7 verdict logic is tested on exact arithmetic
rather than on whatever MiniLM outputs for a given sentence -- which is neither
stable across versions nor a statement about the code under test.

The model is pinned by revision in models.lock like every other weight here. An
unpinned embedding model silently changes explanation-correctness scores between
runs.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Correctness — three signals and the conjunctive verdict

**Files:**
- Create: `src/xai_vlm/metrics/correctness.py`
- Test: `tests/test_correctness.py`

**Interfaces:**
- Consumes: `geometry.max_iou`, `geometry.to_normalised`, `text_similarity.Embedder`, `text_similarity.cosine`.
- Produces:
  - `SIMILARITY_FLOOR = 0.60`, `IOU_FLOOR = 0.30`
  - `HAZARD_DESCRIPTIONS: dict[str, str]`
  - `def hazard_match(rationale_vec, asked_rule: str, hazard_vecs: dict) -> bool`
  - `def classify(similarity: float, hazard_ok: bool, iou_value: float) -> str` — `"CORRECT"` / `"INCORRECT"`
  - `def is_undefined(verdict: str, region: str, human_boxes: list, image_w: int) -> str` — reason string or `""`

- [ ] **Step 1: Write the failing test**

```python
# studies/02-vlm-xai-study/tests/test_correctness.py
"""Metric 7 verdict logic, on exact arithmetic via a fake embedder."""
import pytest

from xai_vlm.metrics.correctness import (
    IOU_FLOOR,
    SIMILARITY_FLOOR,
    classify,
    is_undefined,
)


def test_thresholds_are_the_pre_registered_values():
    """These were fixed before any value was computed. Changing them is a
    deliberate, visible edit -- never a quiet slide to fit a result."""
    assert SIMILARITY_FLOOR == 0.60
    assert IOU_FLOOR == 0.30


def test_the_architecture_docs_correct_example():
    """sim 0.88, PPE match, IoU 0.63 -> CORRECT (arch doc Stage 10)."""
    assert classify(0.88, True, 0.63) == "CORRECT"


def test_the_architecture_docs_incorrect_example():
    """sim 0.11, no hazard match, IoU 0.03 -> INCORRECT (arch doc Stage 10)."""
    assert classify(0.11, False, 0.03) == "INCORRECT"


def test_all_three_signals_are_required():
    assert classify(0.95, True, 0.10) == "INCORRECT"   # points at the wrong place
    assert classify(0.10, True, 0.95) == "INCORRECT"   # says the wrong thing
    assert classify(0.95, False, 0.95) == "INCORRECT"  # wrong hazard category


def test_a_plausible_sentence_pointing_elsewhere_fails():
    """The spatial signal is the anchor. High text similarity must not rescue a
    region on the wrong side of the photograph."""
    assert classify(0.99, True, 0.0) == "INCORRECT"


def test_boundary_values_pass():
    assert classify(SIMILARITY_FLOOR, True, IOU_FLOOR) == "CORRECT"


def test_undefined_when_the_model_said_no():
    assert is_undefined("NO", "(1,2,3,4)", [(0.1, 0.1, 0.2, 0.2)], 1200) == "verdict_no"


def test_undefined_when_no_region_was_given():
    assert is_undefined("YES", "", [(0.1, 0.1, 0.2, 0.2)], 1200) == "no_region"


def test_undefined_when_the_annotation_has_no_box():
    assert is_undefined("YES", "(1,2,3,4)", [], 1200) == "no_annotation_box"


def test_undefined_when_image_dimensions_are_unknown():
    """Without dimensions the model box cannot be normalised, so IoU is undefined
    rather than zero."""
    assert is_undefined("YES", "(1,2,3,4)", [(0.1, 0.1, 0.2, 0.2)], 0) == "no_dimensions"


def test_defined_rows_return_empty_string():
    assert is_undefined("YES", "(1,2,3,4)", [(0.1, 0.1, 0.2, 0.2)], 1200) == ""
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_correctness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xai_vlm.metrics.correctness'`

- [ ] **Step 3: Write the implementation**

```python
# studies/02-vlm-xai-study/src/xai_vlm/metrics/correctness.py
"""Stage 10 -- Metric 7: Explanation Correctness.

Does the model's stated reason match the human's? This is the only metric in the
battery that can catch a RIGHT ANSWER FOR THE WRONG REASON -- and after Stage 2
measured 91% sensitivity against 41% specificity, that is the live question. A
model answering YES to 70% of everything gets many verdicts right by disposition,
and no accuracy measure can tell disposition from understanding.
"""

from __future__ import annotations

import numpy as np

from xai_vlm.metrics.text_similarity import cosine
from xai_vlm.rules import RULES

# PRE-REGISTERED, fixed before any value was computed, taken from convention:
# 0.60 is a standard paraphrase floor for MiniLM-class bi-encoders; 0.30 is a
# deliberately permissive IoU, because the model is asked for ONE region while the
# annotation may mark several.
#
# NOT TO BE TUNED AGAINST OUR RESULTS. All three raw signals are written to the
# output CSV, so anyone wanting a different threshold can re-derive without
# re-running anything -- which is the honest way to explore sensitivity.
SIMILARITY_FLOOR = 0.60
IOU_FLOOR = 0.30

# One canonical sentence per hazard, used to classify what the model's rationale
# is actually about. Phrased as the dataset would describe the hazard, not as the
# question was asked, so a rationale is matched on content rather than on echoing
# the prompt back.
HAZARD_DESCRIPTIONS = {
    "rule_1": "A worker is not wearing required protective equipment such as a "
              "hard hat or high-visibility clothing.",
    "rule_2": "A worker is at height without fall protection such as a harness "
              "or lanyard.",
    "rule_3": "There is an unguarded edge, excavation, trench or floor opening "
              "someone could fall into.",
    "rule_4": "A worker is too close to operating heavy machinery such as an "
              "excavator.",
}


def hazard_match(rationale_vec: np.ndarray, asked_rule: str,
                 hazard_vecs: dict) -> bool:
    """Does the rationale describe the hazard that was asked about?

    Catches answering YES about PPE while describing a trench -- a gross category
    error that text similarity alone can miss when both sentences are about
    construction sites.
    """
    best, best_sim = None, -2.0
    for rule_id, vec in hazard_vecs.items():
        sim = cosine(rationale_vec, vec)
        if sim > best_sim:
            best, best_sim = rule_id, sim
    return best == asked_rule


def classify(similarity: float, hazard_ok: bool, iou_value: float) -> str:
    """Conjunctive verdict: all three signals must pass.

    Conjunctive rather than weighted because the SPATIAL SIGNAL IS THE ANCHOR.
    Text similarity is fuzzy; box overlap is arithmetic. A weighted score would let
    a high text score outvote a region on the wrong side of the photograph, which
    is the exact failure geometry exists to catch.
    """
    ok = (similarity >= SIMILARITY_FLOOR) and hazard_ok and (iou_value >= IOU_FLOOR)
    return "CORRECT" if ok else "INCORRECT"


def is_undefined(verdict: str, region: str, human_boxes: list, image_w: int) -> str:
    """Why this row cannot be scored, or "" if it can.

    Undefined is NOT incorrect. Scoring these as INCORRECT would inflate the
    failure rate with cases the metric does not cover; scoring them CORRECT would
    do the reverse. Either is a silent lie about coverage.
    """
    if verdict != "YES":
        return "verdict_no"
    if not (region or "").strip():
        return "no_region"
    if not human_boxes:
        return "no_annotation_box"
    if not image_w:
        return "no_dimensions"
    return ""


def hazard_of(rule_id: str) -> str:
    """The hazard category a rule belongs to."""
    return RULES[rule_id].hazard
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_correctness.py -v`
Expected: 11 passed

- [ ] **Step 5: Add the hazard-match test with the fake embedder**

```python
# append to studies/02-vlm-xai-study/tests/test_correctness.py
import numpy as np

from xai_vlm.metrics.correctness import HAZARD_DESCRIPTIONS, hazard_match


def test_hazard_match_picks_the_nearest_hazard():
    """A rationale about a trench must not pass as a PPE answer."""
    vecs = {
        "rule_1": np.array([1.0, 0.0, 0.0, 0.0]),
        "rule_2": np.array([0.0, 1.0, 0.0, 0.0]),
        "rule_3": np.array([0.0, 0.0, 1.0, 0.0]),
        "rule_4": np.array([0.0, 0.0, 0.0, 1.0]),
    }
    ppe_like = np.array([0.9, 0.1, 0.0, 0.0])
    trench_like = np.array([0.0, 0.0, 0.9, 0.1])

    assert hazard_match(ppe_like, "rule_1", vecs) is True
    assert hazard_match(trench_like, "rule_1", vecs) is False
    assert hazard_match(trench_like, "rule_3", vecs) is True


def test_every_rule_has_a_hazard_description():
    from xai_vlm.rules import RULE_IDS
    assert set(HAZARD_DESCRIPTIONS) == set(RULE_IDS)
```

- [ ] **Step 6: Run to verify it passes**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_correctness.py -v`
Expected: 13 passed

- [ ] **Step 7: Commit**

```bash
git add studies/02-vlm-xai-study/src/xai_vlm/metrics/correctness.py studies/02-vlm-xai-study/tests/test_correctness.py
git commit -m "feat(study2): metric 7 verdict logic, conjunctive and pre-registered

All three signals must pass: similarity >= 0.60, hazard match, IoU >= 0.30. The
spatial signal is the anchor -- a weighted score would let a high text score
outvote a region on the wrong side of the photograph, which is the exact failure
geometry exists to catch.

The architecture document's two worked examples are literal test cases.

Undefined rows -- NO verdict, no region, no annotation box, no dimensions -- are
classified with a reason and excluded, never coerced. Scoring them INCORRECT would
inflate the failure rate with cases the metric does not cover.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Stage 10 script and the headline

**Files:**
- Create: `scripts/10_explanation_correctness.py`
- Test: `tests/test_explanation_correctness.py`

**Interfaces:**
- Consumes: everything from Tasks 1, 2, 4, 5.
- Produces:
  - `RESULT_FIELDS: list[str]`
  - `def score_rows(baseline, regions, annotations, embedder) -> list[dict]`
  - `def summarise(rows: list[dict]) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# studies/02-vlm-xai-study/tests/test_explanation_correctness.py
"""Stage 10 end to end, over a fake embedder. No model weights."""
import importlib.util
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm.metrics.text_similarity import FakeEmbedder  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "explanation_correctness", STUDY_ROOT / "scripts" / "10_explanation_correctness.py")
stage10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage10)

HUMAN = "The mechanic in front of the truck is not wearing a hard hat."
MODEL_GOOD = "The man working beside the truck cab has no helmet on."
MODEL_BAD = "A worker is not wearing a high-visibility vest."


def _embedder():
    # Good model text sits near the human text and near rule_1's description;
    # bad text is orthogonal to the human text.
    return FakeEmbedder({
        HUMAN:       [1.0, 0.0, 0.0, 0.0],
        MODEL_GOOD:  [0.98, 0.2, 0.0, 0.0],
        MODEL_BAD:   [0.0, 0.0, 0.0, 1.0],
        stage10.HAZARD_DESCRIPTIONS["rule_1"]: [1.0, 0.0, 0.0, 0.0],
        stage10.HAZARD_DESCRIPTIONS["rule_2"]: [0.0, 1.0, 0.0, 0.0],
        stage10.HAZARD_DESCRIPTIONS["rule_3"]: [0.0, 0.0, 1.0, 0.0],
        stage10.HAZARD_DESCRIPTIONS["rule_4"]: [0.0, 0.0, 0.0, 1.0],
    })


def _row(image_id, rationale, verdict="YES", region="(600.0, 450.0, 720.0, 540.0)"):
    return {"image_id": image_id, "rule_id": "rule_1", "verdict": verdict,
            "rationale": rationale, "region": region}


def _regions(image_id, region="(600.0, 450.0, 720.0, 540.0)", w=1200, h=900):
    return {"image_id": image_id, "rule_id": "rule_1", "verdict": "YES",
            "source": "model" if region else "none", "region": region,
            "image_w": w, "image_h": h}


def _ann(image_id, boxes, reason=HUMAN, quality="rich info"):
    return {image_id: {"rule_1": {"bounding_box": boxes, "reason": reason},
                       "quality_of_info": quality}}


def test_a_good_explanation_scores_correct():
    boxes = [[0.5, 0.5, 0.6, 0.6]]
    rows = stage10.score_rows([_row("0001", MODEL_GOOD)], [_regions("0001")],
                              _ann("0001", boxes), _embedder())
    assert rows[0]["verdict_correctness"] == "CORRECT"
    assert rows[0]["undefined_reason"] == ""


def test_right_answer_wrong_reason_is_caught():
    """The model says YES, ground truth is YES, but it describes another hazard
    in a different part of the frame."""
    boxes = [[0.0, 0.0, 0.1, 0.1]]
    rows = stage10.score_rows([_row("0002", MODEL_BAD)], [_regions("0002")],
                              _ann("0002", boxes), _embedder())
    assert rows[0]["verdict_correctness"] == "INCORRECT"


def test_no_verdict_rows_are_undefined_not_incorrect():
    rows = stage10.score_rows([_row("0003", MODEL_GOOD, verdict="NO")],
                              [_regions("0003")], _ann("0003", [[0.5, 0.5, 0.6, 0.6]]),
                              _embedder())
    assert rows[0]["undefined_reason"] == "verdict_no"
    assert rows[0]["verdict_correctness"] == ""


def test_summary_reports_undefined_separately_from_incorrect():
    rows = [
        {"verdict_correctness": "CORRECT", "undefined_reason": "", "quality_of_info": "rich info"},
        {"verdict_correctness": "INCORRECT", "undefined_reason": "", "quality_of_info": "rich info"},
        {"verdict_correctness": "", "undefined_reason": "no_region", "quality_of_info": "poor info"},
    ]
    s = stage10.summarise(rows)
    assert s["n_scored"] == 2
    assert s["n_undefined"] == 1
    assert s["correct_rate"] == 0.5


def test_coordinates_are_converted_before_comparison():
    """The model box in pixels covers the image centre; the annotation in
    normalised units covers the same place. Without conversion IoU would be 0."""
    boxes = [[0.5, 0.5, 0.6, 0.6]]
    rows = stage10.score_rows([_row("0004", MODEL_GOOD)], [_regions("0004")],
                              _ann("0004", boxes), _embedder())
    assert rows[0]["iou"] > 0.9
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_explanation_correctness.py -v`
Expected: FAIL — the module does not exist yet.

- [ ] **Step 3: Write the stage script**

```python
# studies/02-vlm-xai-study/scripts/10_explanation_correctness.py
"""Stage 10 -- Metric 7: Explanation Correctness.

Does the model's stated reason match the human's? The headline is the
RIGHT-ANSWER-WRONG-REASON RATE: cases where the verdict was correct and the
explanation was not. It is the most important failure mode in safety-critical AI
and is invisible to every other metric in this battery.

    python scripts/10_explanation_correctness.py --model qwen7b-4bit
"""
import argparse
import ast
import csv
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402
from xai_vlm.metrics.correctness import (  # noqa: E402
    HAZARD_DESCRIPTIONS,
    classify,
    hazard_match,
    is_undefined,
)
from xai_vlm.metrics.geometry import max_iou, to_normalised  # noqa: E402
from xai_vlm.metrics.text_similarity import MiniLMEmbedder, cosine  # noqa: E402

RESULT_FIELDS = [
    "image_id", "rule_id", "verdict", "expected", "quality_of_info",
    "similarity", "hazard_ok", "iou", "verdict_correctness",
    "undefined_reason", "model_reason", "human_reason",
]


def _parse_box(text: str):
    """'(x0, y0, x1, y1)' -> tuple, or None."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        v = ast.literal_eval(text)
        return tuple(float(x) for x in v) if len(v) == 4 else None
    except (ValueError, SyntaxError):
        return None


def score_rows(baseline, regions, annotations, embedder) -> list[dict]:
    """One scored row per baseline row."""
    reg_by_key = {(r["image_id"], r["rule_id"]): r for r in regions}

    hazard_vecs = dict(zip(
        HAZARD_DESCRIPTIONS,
        embedder.encode([HAZARD_DESCRIPTIONS[k] for k in HAZARD_DESCRIPTIONS]),
    ))

    out = []
    for row in baseline:
        key = (row["image_id"], row["rule_id"])
        reg = reg_by_key.get(key, {})
        ann = annotations.get(row["image_id"], {})
        rule_ann = (ann.get(row["rule_id"]) or {}) if ann else {}
        human_boxes = rule_ann.get("bounding_box") or []
        human_reason = rule_ann.get("reason", "") or ""
        image_w = int(reg.get("image_w") or 0)
        image_h = int(reg.get("image_h") or 0)

        reason = is_undefined(row.get("verdict", ""), row.get("region", ""),
                              human_boxes, image_w)

        rec = {
            "image_id": row["image_id"], "rule_id": row["rule_id"],
            "verdict": row.get("verdict", ""),
            "expected": row.get("expected", ""),
            "quality_of_info": ann.get("quality_of_info", "") if ann else "",
            "similarity": "", "hazard_ok": "", "iou": "",
            "verdict_correctness": "", "undefined_reason": reason,
            "model_reason": (row.get("rationale") or "")[:200],
            "human_reason": human_reason[:200],
        }

        if not reason:
            vecs = embedder.encode([row.get("rationale") or "", human_reason])
            sim = cosine(vecs[0], vecs[1])
            hz = hazard_match(vecs[0], row["rule_id"], hazard_vecs)
            model_box = _parse_box(row.get("region", ""))
            iou_v = 0.0
            if model_box is not None:
                iou_v = max_iou(to_normalised(model_box, image_w, image_h),
                                [tuple(b) for b in human_boxes])
            rec["similarity"] = round(sim, 4)
            rec["hazard_ok"] = hz
            rec["iou"] = round(iou_v, 4)
            rec["verdict_correctness"] = classify(sim, hz, iou_v)

        out.append(rec)
    return out


def summarise(rows: list[dict]) -> dict:
    """Scored versus undefined, kept strictly apart."""
    scored = [r for r in rows if not r["undefined_reason"]]
    undefined = [r for r in rows if r["undefined_reason"]]
    correct = [r for r in scored if r["verdict_correctness"] == "CORRECT"]
    return {
        "n_total": len(rows),
        "n_scored": len(scored),
        "n_undefined": len(undefined),
        "correct_rate": len(correct) / len(scored) if scored else float("nan"),
    }


def load_annotations(loader, wanted: set) -> dict:
    """image_id -> {rule_id: {bounding_box, reason}, quality_of_info}."""
    from xai_vlm.rules import RULE_IDS
    out = {}
    for split in config.SPLITS:
        ds = loader(split=split, streaming=True)
        if hasattr(ds, "remove_columns"):
            ds = ds.remove_columns(["image"])
        for rec in ds:
            iid = rec["image_id"]
            if iid not in wanted or iid in out:
                continue
            entry = {"quality_of_info": rec.get("quality_of_info", "")}
            for rid in RULE_IDS:
                v = rec.get(f"{rid}_violation")
                if v:
                    entry[rid] = {"bounding_box": v.get("bounding_box") or [],
                                  "reason": v.get("reason", "") or ""}
            out[iid] = entry
            if len(out) >= len(wanted):
                return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen7b-4bit")
    args = ap.parse_args()

    res = STUDY_ROOT / "results"
    with (res / f"baseline_{args.model}.csv").open(newline="", encoding="utf-8") as fh:
        baseline = list(csv.DictReader(fh))
    regions_path = res / f"regions_{args.model}.csv"
    if not regions_path.exists():
        print(f"no {regions_path}. Run scripts/03_regions.py first.", file=sys.stderr)
        return 1
    with regions_path.open(newline="", encoding="utf-8") as fh:
        regions = list(csv.DictReader(fh))

    from xai_pilot.data import load_construction_site

    print("loading annotations ...")
    annotations = load_annotations(load_construction_site,
                                   {r["image_id"] for r in baseline})
    print(f"  {len(annotations)} images")

    print("scoring ...")
    rows = score_rows(baseline, regions, annotations, MiniLMEmbedder())

    out = res / f"explanation_correctness_{args.model}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    s = summarise(rows)
    raww = [r for r in rows if not r["undefined_reason"]
            and r["verdict"] == "YES" and r["expected"] == "YES"
            and r["verdict_correctness"] == "INCORRECT"]
    right_answers = [r for r in rows if not r["undefined_reason"]
                     and r["verdict"] == "YES" and r["expected"] == "YES"]

    print("\n" + "=" * 66)
    print(f"scored {s['n_scored']} of {s['n_total']}   "
          f"undefined {s['n_undefined']}")
    from collections import Counter
    for reason, n in Counter(r["undefined_reason"] for r in rows
                             if r["undefined_reason"]).most_common():
        print(f"    {n:>5}  {reason}")
    print(f"\nexplanation CORRECT rate : {s['correct_rate']:.1%}")
    if right_answers:
        print(f"RIGHT ANSWER, WRONG REASON : "
              f"{len(raww) / len(right_answers):.1%} "
              f"({len(raww)}/{len(right_answers)})")
        print("  the model was right, and could not say why correctly")
    print("\nsplit by quality_of_info:")
    for q in sorted({r["quality_of_info"] for r in rows if r["quality_of_info"]}):
        sub = [r for r in rows if r["quality_of_info"] == q and not r["undefined_reason"]]
        if sub:
            ok = sum(1 for r in sub if r["verdict_correctness"] == "CORRECT")
            print(f"  {q:<12} {ok / len(sub):.1%}  (n={len(sub)})")
    print(f"\nwrote {out}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../../pilot/.venv/Scripts/python -m pytest tests/test_explanation_correctness.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the whole suite**

Run: `../../pilot/.venv/Scripts/python -m pytest tests -q`
Expected: all pass, 12 skipped (GPU).

- [ ] **Step 6: Run Stage 10 for real**

Run: `../../pilot/.venv/Scripts/python scripts/10_explanation_correctness.py --model qwen7b-4bit`

Expected: a correctness rate, a right-answer-wrong-reason rate, an undefined breakdown, and a `quality_of_info` split. **Sanity-check before believing it:** if the correct rate is 0.0% or 100.0%, or if IoU is zero for essentially every row, stop — those are the signatures of the coordinate bug, not findings.

- [ ] **Step 7: Commit**

```bash
git add studies/02-vlm-xai-study/scripts/10_explanation_correctness.py studies/02-vlm-xai-study/tests/test_explanation_correctness.py studies/02-vlm-xai-study/results/explanation_correctness_qwen7b-4bit.csv
git commit -m "feat(study2): Stage 10 -- explanation correctness

Headline is the right-answer-wrong-reason rate: the verdict was correct and the
explanation was not. After Stage 2 measured 91% sensitivity against 41%
specificity, that is the live question -- a model answering YES to 70% of
everything gets many verdicts right by disposition, and no accuracy measure can
tell disposition from understanding.

All three raw signals are written beside the verdict so any reader can
re-threshold without re-running. Results split by quality_of_info.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Documentation

**Files:**
- Modify: `studies/02-vlm-xai-study/README.md`

- [ ] **Step 1: Update the status table**

Add rows for `metrics/geometry.py`, `regions.py`, `efficiency.py`, `text_similarity.py`, `correctness.py`, and Stages 3, 7, 10 as implemented and run. Update the test count to the number the suite actually reports.

- [ ] **Step 2: Add a "Derived metric results" section**

Record the no-region rate, the efficiency figures with the in-run/standalone caveat, the explanation-correctness rate, the right-answer-wrong-reason rate, the undefined breakdown, and the `quality_of_info` split — with the exact numbers the runs produced.

- [ ] **Step 3: Record the pre-registered thresholds**

State `SIMILARITY_FLOOR = 0.60` and `IOU_FLOOR = 0.30`, that they were fixed before any value was computed, and that all three raw signals are in the CSV so re-thresholding needs no re-run.

- [ ] **Step 4: Commit**

```bash
git add studies/02-vlm-xai-study/README.md
git commit -m "docs(study2): record Plan 3a results and pre-registered thresholds

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Plan Self-Review

**Spec coverage.** Stage 3 → Task 2; Stage 7 → Task 3; Stage 10 → Tasks 1, 4, 5, 6. Coordinate conversion → Task 1 (with the top-left-corner regression test). Max-over-boxes → Task 1. Pinned embedding model → Task 4. Conjunctive verdict and pre-registered thresholds → Task 5. Undefined-not-incorrect → Task 5 and Task 6. `quality_of_info` split → Task 6. The §6.3 worked examples → Task 5. Real-model verification → Task 4 Step 6.

**Deviation from the spec, deliberate.** The spec put `to_normalised` and IoU inside `correctness.py`; the plan gives them their own `geometry.py`. Box arithmetic is the one part with a named, documented failure mode, and it deserves its own test file rather than sharing one with verdict logic.

**Interface consistency.** `Box`, `to_normalised`, `iou`, `max_iou`, `classify_source`, `build_region_rows`, `REGION_FIELDS`, `latency_stats`, `per_rule_stats`, `Embedder`, `FakeEmbedder`, `MiniLMEmbedder`, `cosine`, `SIMILARITY_FLOOR`, `IOU_FLOOR`, `HAZARD_DESCRIPTIONS`, `hazard_match`, `classify`, `is_undefined`, `score_rows`, `summarise`, `RESULT_FIELDS` are each defined once and referenced with the same names throughout.

**Test counts** in the run steps are exact per-file expectations; the suite-wide count is not asserted, because it moves as tasks land. Report the actual number.
