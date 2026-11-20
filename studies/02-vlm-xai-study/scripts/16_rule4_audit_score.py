"""Rule_4 audit -- score it, and correct both tiers' rule_4 confusion matrix.

Each drawn row was relabelled blind (yes / no / unsure: is a worker in the
blind spot or operating radius of an operating excavator). Per draw cell, the
share of answered rows the reader calls a violation, p_cell, is that cell's
estimated true-violation rate. Every one of the 323 rule_4 rows is then given
its cell's p, and each tier's confusion matrix is recomputed in expectation:

    TP = sum of p over rows the tier answered YES     FP = sum of (1 - p) there
    FN = sum of p over rows the tier answered NO      TN = sum of (1 - p) there

beside the same figures with p = 1 for annotated rows and 0 otherwise. The
bootstrap resamples answered rows within each cell. "unsure" is excluded from
p and reported per cell.

    python scripts/16_rule4_audit_score.py ANSWERS_DIR     # ANSWERS_DIR/<image_id>.json
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402

TIERS = ("qwen7b-4bit", "qwen7b-bf16")


def _yes(v) -> bool:
    return str(v).upper() == "YES"


def cell_answers(manifest: list[dict], answers: dict) -> dict:
    """{cell: [True/False per answered row]} plus unsure counts."""
    out: dict = {}
    for m in manifest:
        a = (answers.get(m["image_id"]) or {}).get("rule_4", "")
        c = out.setdefault(m["cell"], {"yn": [], "unsure": 0, "missing": 0})
        if a in ("yes", "no"):
            c["yn"].append(a == "yes")
        elif a == "unsure":
            c["unsure"] += 1
        else:
            c["missing"] += 1
    return out


def confusion(rows: list[dict], p_of, tier: str) -> dict:
    """Expected confusion counts for one tier; p_of(row) is P(true violation)."""
    tp = fp = fn = tn = 0.0
    fa = {"C": [0.0, 0.0], "H": [0.0, 0.0]}  # [false-alarm mass, true-negative mass]
    for r in rows:
        p = p_of(r)
        if _yes(r[tier]):
            tp += p
            fp += 1 - p
        else:
            fn += p
            tn += 1 - p
        s = "C" if r["stratum"] == "C" else "H"
        if not _yes(r["expected"]):
            fa[s][0 if _yes(r[tier]) else 1] += 1 - p
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return {"sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "false_alarm_C": div(fa["C"][0], sum(fa["C"])),
            "false_alarm_H": div(fa["H"][0], sum(fa["H"])),
            "violations": tp + fn}


def annotation_p(row: dict) -> float:
    return 1.0 if _yes(row["expected"]) else 0.0


def audit_p(cells: dict):
    p = {c: sum(v["yn"]) / len(v["yn"]) for c, v in cells.items() if v["yn"]}
    return lambda row: p[row["cell"]]


def bootstrap(rows, cells, tier, n_boot=5000, seed=0):
    rng = random.Random(seed)
    keys = ("sensitivity", "specificity", "false_alarm_C", "false_alarm_H")
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        bc = {c: {"yn": [rng.choice(v["yn"]) for _ in v["yn"]]} for c, v in cells.items() if v["yn"]}
        res = confusion(rows, audit_p(bc), tier)
        for k in keys:
            draws[k].append(res[k])
    out = {}
    for k, xs in draws.items():
        xs = sorted(x for x in xs if x == x)
        out[k] = (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]) if xs else (float("nan"),) * 2
    return out


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("answers_dir")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    spec = __import__("importlib.util").util.spec_from_file_location(
        "s15", STUDY_ROOT / "scripts" / "15_rule4_audit_sample.py")
    s15 = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(s15)
    rows = s15.rule4_rows({t: _read(config.RESULTS_DIR / "scholar-a40" / f"baseline_{t}.csv")
                           for t in TIERS})
    manifest = _read(config.RESULTS_DIR / "rule4_audit_manifest.csv")
    answers = {}
    for f in sorted(Path(args.answers_dir).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        d = d.get("data", d) if isinstance(d.get("data"), dict) else d
        answers[d.get("image_id") or f.stem] = d
    cells = cell_answers(manifest, answers)

    print(f"rule_4 rows in the sample: {len(rows)}; drawn {len(manifest)}; answered "
          f"{sum(len(v['yn']) for v in cells.values())} yes/no, "
          f"{sum(v['unsure'] for v in cells.values())} unsure, "
          f"{sum(v['missing'] for v in cells.values())} missing")
    print("\ncell   size  drawn  reader-yes / answered  unsure")
    sizes = {m["cell"]: int(m["cell_size"]) for m in manifest}
    for c in s15.PER_CELL:
        v = cells.get(c, {"yn": [], "unsure": 0})
        print(f"  {c:<5} {sizes.get(c, 0):>4}  {len(v['yn']) + v['unsure']:>5}  "
              f"{sum(v['yn']):>6} / {len(v['yn']):<3}            {v['unsure']}")
    if any(not v["yn"] for v in cells.values()):
        print("\na cell has no yes/no answers; corrected figures need every cell", file=sys.stderr)
        return 1
    for tier in TIERS:
        a = confusion(rows, annotation_p, tier)
        b = confusion(rows, audit_p(cells), tier)
        ci = bootstrap(rows, cells, tier, n_boot=args.n_boot)
        print(f"\n== {tier} (rule_4, {len(rows)} rows)   annotation  ->  audit-corrected [95% CI]")
        for k in ("sensitivity", "specificity", "false_alarm_C", "false_alarm_H"):
            lo, hi = ci[k]
            print(f"  {k:<15} {a[k]:7.1%}  ->  {b[k]:6.1%} [{lo:.1%}, {hi:.1%}]")
        print(f"  violations      {a['violations']:7.1f}  ->  {b['violations']:6.1f} (expected count)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
