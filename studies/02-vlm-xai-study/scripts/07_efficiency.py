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
