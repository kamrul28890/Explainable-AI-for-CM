"""Roster step 3 (design D5) -- how far does the library change alone move replies?

The same tier, the same rows and the same A40, answered in environment A
(transformers 4.49, every frozen result) and environment B (current transformers,
needed by the newer models). Nothing else differs. This is a measurement, not a
pass/fail check: the paper states the size of the shift.

    python scripts/21_env_parity.py \
        --a results/scholar-a40-rf/baseline_qwen7b-bf16.csv \
        --b results/scholar-a40-rf/baseline_qwen7b-bf16_envB-parity.csv
"""
import argparse
import ast
import csv
import sys
from pathlib import Path

csv.field_size_limit(10 ** 8)


def _box(text: str):
    text = (text or "").strip()
    return tuple(float(v) for v in ast.literal_eval(text)) if text else None


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Agreement between two runs of the same rows, keyed on (image_id, rule_id)."""
    a = {(r["image_id"], r["rule_id"]): r for r in rows_a}
    b = {(r["image_id"], r["rule_id"]): r for r in rows_b}
    keys = sorted(set(a) & set(b))
    out = {"rows": len(keys), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)),
           "raw_identical": 0, "verdict_agree": 0, "region_both": 0, "region_one": 0,
           "region_neither": 0, "region_iou": [], "correct_a": 0, "correct_b": 0}
    for k in keys:
        ra, rb = a[k], b[k]
        out["raw_identical"] += ra["raw_text"] == rb["raw_text"]
        out["verdict_agree"] += ra["verdict"] == rb["verdict"]
        out["correct_a"] += ra["verdict"] == ra["expected"]
        out["correct_b"] += rb["verdict"] == rb["expected"]
        ba, bb = _box(ra["region"]), _box(rb["region"])
        if ba and bb:
            out["region_both"] += 1
            out["region_iou"].append(iou(ba, bb))
        elif ba or bb:
            out["region_one"] += 1
        else:
            out["region_neither"] += 1
    return out


def _read(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="Stage 2 CSV from environment A")
    ap.add_argument("--b", required=True, help="Stage 2 CSV from environment B")
    args = ap.parse_args(argv)
    r = compare(_read(args.a), _read(args.b))
    n = r["rows"] or 1
    ious = sorted(r["region_iou"])
    med = ious[len(ious) // 2] if ious else float("nan")
    print(f"rows compared {r['rows']}  (only in A {r['only_a']}, only in B {r['only_b']})")
    print(f"reply identical      {r['raw_identical']:>4}  {r['raw_identical'] / n:.1%}")
    print(f"verdict agrees       {r['verdict_agree']:>4}  {r['verdict_agree'] / n:.1%}")
    print(f"correct verdict      A {r['correct_a'] / n:.1%}   B {r['correct_b'] / n:.1%}")
    print(f"region in both {r['region_both']}, one {r['region_one']}, neither {r['region_neither']}")
    if ious:
        print(f"A-B region IoU       median {med:.3f}  "
              f">=0.9 {sum(v >= 0.9 for v in ious) / len(ious):.1%}  "
              f">=0.5 {sum(v >= 0.5 for v in ious) / len(ious):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
