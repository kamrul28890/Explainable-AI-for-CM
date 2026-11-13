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
from collections import Counter
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

    res = config.RESULTS_DIR
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
    print(f"scored {s['n_scored']} of {s['n_total']}   undefined {s['n_undefined']}")
    for reason, n in Counter(r["undefined_reason"] for r in rows
                             if r["undefined_reason"]).most_common():
        print(f"    {n:>5}  {reason}")
    print(f"\nexplanation CORRECT rate : {s['correct_rate']:.1%}")
    if right_answers:
        print(f"RIGHT ANSWER, WRONG REASON : "
              f"{len(raww) / len(right_answers):.1%} "
              f"({len(raww)}/{len(right_answers)})")
        print("  the model was right, and could not say why correctly")

    scored = [r for r in rows if not r["undefined_reason"]]
    if scored:
        print("\nsignal pass rates among scored rows:")
        from xai_vlm.metrics.correctness import IOU_FLOOR, SIMILARITY_FLOOR
        print(f"  similarity >= {SIMILARITY_FLOOR} : "
              f"{sum(1 for r in scored if float(r['similarity']) >= SIMILARITY_FLOOR) / len(scored):.1%}")
        print(f"  hazard match          : "
              f"{sum(1 for r in scored if r['hazard_ok'] is True or r['hazard_ok'] == 'True') / len(scored):.1%}")
        print(f"  IoU >= {IOU_FLOOR}          : "
              f"{sum(1 for r in scored if float(r['iou']) >= IOU_FLOOR) / len(scored):.1%}")

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
