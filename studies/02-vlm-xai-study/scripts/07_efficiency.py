"""Stage 7 -- Metric 4: Efficiency, from timings already recorded.

    python scripts/07_efficiency.py --model qwen7b-4bit
"""
import argparse
import csv
import json
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402

from xai_vlm.metrics.efficiency import latency_stats, per_rule_stats  # noqa: E402

# Measured standalone on the same images, for the caveat below. See the Plan 3a
# design doc and runner.py: on the local machine identical work cost 3.1 s alone
# and ~10 s inside a run. That standalone figure belongs to that GPU only.
STANDALONE_MEDIAN_S = 3.1
STANDALONE_DEVICE = "NVIDIA GeForce RTX 3070"   # torch.cuda.get_device_name() here


def run_device(results_dir: Path, model: str) -> str:
    """The GPU that produced this tree's baseline, from its Stage 2 fingerprint.

    State files written before 2026-11-09 have no `device` key; every one of them
    is the frozen local run, so a missing key names the local GPU."""
    path = results_dir / f"baseline_{model}.state.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("device") or STANDALONE_DEVICE
    except FileNotFoundError:
        return "unknown"


def latency_caveat(in_run_median_s: float, device: str) -> list[str]:
    """The caveat printed with every efficiency figure.

    A standalone/in-run ratio is quoted only on the GPU that measured the
    standalone figure -- dividing an A40 latency by a 3070 latency measures the
    hardware, not the harness."""
    lines = [
        "CAVEAT, reported with every efficiency figure:",
        f"  in-run median      {in_run_median_s:.1f}s on {device}   <- PRIMARY, what a batch costs",
        f"  standalone median  {STANDALONE_MEDIAN_S:.1f}s on {STANDALONE_DEVICE}",
    ]
    if device == STANDALONE_DEVICE:
        lines.append(f"  in a run the same images take {in_run_median_s / STANDALONE_MEDIAN_S:.1f}x "
                     "longer than alone on this machine.")
    else:
        lines.append("  different GPUs: no standalone/in-run ratio is meaningful here.")
    lines += [
        "  The gap is a property of the local machine, not of the harness: the same",
        "  code and data ran at 2.3s per call in-run on an A40 (results/scholar-a40,",
        "  2026-11-09). Compare latencies only within one machine's results tree.",
    ]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen7b-4bit")
    args = ap.parse_args()

    path = config.RESULTS_DIR / f"baseline_{args.model}.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    lat = [float(r["latency_ms"]) / 1000.0 for r in rows if (r.get("latency_ms") or "").strip()]
    overall = latency_stats(lat)
    by_rule = per_rule_stats(rows)

    out = config.RESULTS_DIR / f"efficiency_{args.model}.csv"
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
    for line in latency_caveat(overall["median"], run_device(config.RESULTS_DIR, args.model)):
        print(line)
    print(f"wrote {out}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
