"""Stage 9 -- Metric 6: Bounded Completeness. No new model calls.

Joins Stage 3 (was an explanation offered?) with Stage 4 (did it matter?).

    python scripts/09_completeness.py --model qwen7b-4bit
"""
import argparse
import csv
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm.metrics.completeness import (  # noqa: E402
    CATEGORIES,
    classify_completeness,
    completeness_rates,
)
from xai_vlm.metrics.reporting import rate_row  # noqa: E402

RESULT_FIELDS = ["image_id", "rule_id", "verdict", "source", "flip_tokens",
                 "flip_black", "completeness"]


def build_rows(regions: list[dict], da: list[dict]) -> list[dict]:
    """One completeness row per region row."""
    da_by_key = {(r["image_id"], r["rule_id"]): r for r in da}
    out = []
    for reg in regions:
        key = (reg["image_id"], reg["rule_id"])
        d = da_by_key.get(key, {})
        out.append({
            "image_id": reg["image_id"],
            "rule_id": reg["rule_id"],
            "verdict": reg.get("verdict", ""),
            "source": reg.get("source", ""),
            "flip_tokens": d.get("flip_tokens", ""),
            "flip_black": d.get("flip_black", ""),
            "completeness": classify_completeness(reg.get("source", ""),
                                                  d.get("flip_tokens", "")),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen7b-4bit")
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()

    res = STUDY_ROOT / "results"
    reg_path = res / f"regions_{args.model}.csv"
    da_path = res / f"descriptive_accuracy_{args.model}.csv"
    if not reg_path.exists():
        print(f"missing {reg_path}; run scripts/03_regions.py.", file=sys.stderr)
        return 1
    if not da_path.exists():
        print(f"missing {da_path}; run scripts/04_descriptive_accuracy.py.",
              file=sys.stderr)
        return 1

    with reg_path.open(newline="", encoding="utf-8") as fh:
        regions = list(csv.DictReader(fh))
    with da_path.open(newline="", encoding="utf-8") as fh:
        da = list(csv.DictReader(fh))

    rows = build_rows(regions, da)
    out = res / f"completeness_{args.model}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    rates = completeness_rates(rows)
    print("=" * 70)
    print(f"BOUNDED COMPLETENESS  (n={rates['n']})")
    for cat in CATEGORIES:
        print(f"  {cat:<24} {rates[cat]:.1%}")

    not_measured = sum(1 for r in rows if r["completeness"] == "not_measured")
    if not_measured:
        print(f"\n  {not_measured} rows are not_measured -- Stage 4 has not covered")
        print("  them yet. They are excluded from the headline, never defaulted.")

    measured = [r for r in rows if r["completeness"] != "not_measured"]
    if measured:
        supported = rate_row("explanation_supported",
                             [r["completeness"] == "explanation_supported" for r in measured],
                             args.n_boot)
        weak = rate_row("explanation_weak",
                        [r["completeness"] == "explanation_weak" for r in measured],
                        args.n_boot)
        print(f"\n  among measured rows:")
        print(f"    supported : {supported['formatted']}")
        print(f"    weak      : {weak['formatted']}")
        print("\n  `explanation_weak` is the result to read hardest: the model named")
        print("  a region, the region did not matter, and a reader shown only that")
        print("  region would have been misled about why the system decided.")

    print(f"\nwrote {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
