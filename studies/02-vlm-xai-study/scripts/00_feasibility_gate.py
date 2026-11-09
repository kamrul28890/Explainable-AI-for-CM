"""Stage 0 -- the feasibility gate. Run before building anything else.

Decides whether the study is viable. The decisive question is whether a quantised
model can still detect small hard hats: if it cannot, rule_1 is invalid and so is
most of the study.

Thresholds are fixed in the architecture document BEFORE any numbers are seen.
Do not tune them to fit an observed result -- that is what they exist to prevent.

    python scripts/00_feasibility_gate.py --tier qwen7b-4bit
"""
import argparse
import csv
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402

# PRE-REGISTERED. Not to be adjusted after seeing results.
DETECTION_PROCEED = 0.80
DETECTION_LIMITATION = 0.50
LATENCY_BUDGET_S = 6.0
VRAM_BUDGET_GB = 7.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="qwen7b-4bit")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "stage0_feasibility.csv"))
    args = ap.parse_args()

    from xai_pilot.data import load_construction_site
    from xai_vlm.backends.qwen import QwenBackend

    gate = list(csv.DictReader(
        (STUDY_ROOT / "data" / "gate_images.csv").open(encoding="utf-8")))
    wanted = {g["image_id"] for g in gate}
    images = {}
    for row in load_construction_site(split="test", streaming=True):
        if row["image_id"] in wanted:
            images[row["image_id"]] = row["image"].convert("RGB")
        if len(images) >= len(wanted):
            break

    print(f"loading {args.tier} ...")
    backend = QwenBackend(tier=args.tier)

    rows, latencies = [], []
    for g in gate:
        img = images.get(g["image_id"])
        if img is None:
            continue
        a = backend.answer(img, "rule_1")
        latencies.append(a.latency_ms / 1000.0)
        rows.append({
            "image_id": g["image_id"], "expected": g["expected"],
            "predicted": a.verdict or "UNPARSED", "correct": a.verdict == g["expected"],
            "norm_area": g["norm_area"], "latency_s": round(a.latency_ms / 1000.0, 2),
            "has_region": a.region is not None, "parse_strict": a.parse_strict,
            "rationale": a.rationale[:120], "raw": a.raw_text[:200],
        })
        print(f"  {g['image_id']}  exp={g['expected']}  got={a.verdict}  "
              f"{a.latency_ms/1000:.1f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # --- verdict -----------------------------------------------------------
    small = [r for r in rows if r["norm_area"] and float(r["norm_area"]) < 0.01
             and r["expected"] == "YES"]
    small_rate = sum(r["correct"] for r in small) / len(small) if small else float("nan")
    overall = sum(r["correct"] for r in rows) / len(rows)
    median_latency = sorted(latencies)[len(latencies) // 2]
    parse_rate = sum(1 for r in rows if r["predicted"] != "UNPARSED") / len(rows)
    peak = backend.peak_memory_gb()

    # Sensitivity and specificity separately. small_rate alone cannot distinguish
    # "detects small hazards" from "answers YES to everything" -- a model that
    # always said YES would score 100% on it. The false-alarm rate is what makes
    # the small-object number interpretable, so it is reported beside it.
    pos = [r for r in rows if r["expected"] == "YES"]
    neg = [r for r in rows if r["expected"] == "NO"]
    sens = sum(r["correct"] for r in pos) / len(pos) if pos else float("nan")
    spec = sum(r["correct"] for r in neg) / len(neg) if neg else float("nan")
    yes_share = sum(1 for r in rows if r["predicted"] == "YES") / len(rows)

    print("\n" + "=" * 66)
    print(f"small-object detection : {small_rate:.0%}  (n={len(small)})")
    print(f"overall accuracy       : {overall:.0%}")
    print(f"  sensitivity          : {sens:.0%}  (n={len(pos)})")
    print(f"  specificity          : {spec:.0%}  (n={len(neg)})")
    print(f"  YES share of replies : {yes_share:.0%}   <- 100% would make the")
    print(f"                                        small-object number vacuous")
    print(f"median latency         : {median_latency:.1f}s")
    print(f"parse rate             : {parse_rate:.0%}")
    print(f"peak VRAM              : {peak:.1f} GB")
    print(f"attention available    : {backend.attention(list(images.values())[0], 'rule_1') is not None}")
    print("=" * 66)

    if small_rate != small_rate or small_rate < DETECTION_LIMITATION:
        print("VERDICT: STOP. Raise max_pixels or drop to 3B-8bit and retest.")
        return 1
    if small_rate < DETECTION_PROCEED:
        print("VERDICT: GO, with small-object sensitivity as a stated limitation.")
    else:
        print("VERDICT: GO.")
    if spec == spec and spec < 0.5:
        print(f"NOTE: specificity {spec:.0%} means the small-object result is partly "
              "explained by a YES bias, not only by detection. Report both together, "
              "and treat the false-alarm rate as a headline number (arch doc 5.3).")
    if median_latency > LATENCY_BUDGET_S:
        est = median_latency * 27450 / 3600
        print(f"WARNING: at {median_latency:.1f}s/call the full study is ~{est:.0f}h. "
              "Rescope: robustness subset -> stability subset -> third model. "
              "Never cut Stage 4.")
    if peak > VRAM_BUDGET_GB:
        print(f"WARNING: peak VRAM {peak:.1f} GB is close to the limit; lower max_pixels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
