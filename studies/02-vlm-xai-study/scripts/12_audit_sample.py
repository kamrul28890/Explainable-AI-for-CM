"""Section 5.5 -- draw the 100 images for the human annotation audit.

The dataset grades its own annotations and 47% are "poor info", so annotations
are a reference standard, not ground truth. A person relabels 100 images drawn
50 per quality tier, and the disagreement rate estimates how wrong
the reference is -- reported beside every result.

Drawn from the whole corpus (10,013 images), not the study sample: per quality
tier, 25 images with an annotated violation and 25 without (see `draw`).

    python scripts/12_audit_sample.py                    # writes audit_manifest.csv
    python scripts/12_audit_sample.py --export DIR       # + images and items.json

`--export` writes what the labelling page shows: each image downscaled to a
1024 px long side, its annotations (normalised boxes and reasons), which rules
apply, and -- where the image is in the frozen sample -- the model's Stage 2
answer, shown only after the auditor has answered blind.
"""
import argparse
import ast
import csv
import json
import random
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402
from xai_vlm.rules import RULE_IDS, rule_is_applicable, violated_rules  # noqa: E402

PER_TIER = 50
TIERS = ("rich info", "poor info")
LONG_SIDE = 1024


def draw(records: list[dict], per_tier: int = PER_TIER, seed: int = 42) -> list[dict]:
    """`per_tier` images from each quality tier -- half with an annotated
    violation, half without -- seeded and independent of input order.

    Stratified by hazard (decided 2026-11-10, a change to section 5.5 the paper
    states): the pre-registered pure random draw with seed 42 held 3 hazard
    images in 100, so the rate at which a marked violation is wrong could not be
    estimated. Corpus-level error rates are reweighted to the real prevalence.
    """
    out = []
    for tier in TIERS:
        for hazard in (True, False):
            pool = sorted((r for r in records if r.get("quality_of_info") == tier
                           and bool(violated_rules(r)) == hazard),
                          key=lambda r: r["image_id"])
            rng = random.Random(f"audit-{tier}-{'H' if hazard else 'C'}-{seed}")
            out.extend(rng.sample(pool, per_tier // 2))
    return sorted(out, key=lambda r: (TIERS.index(r["quality_of_info"]), r["image_id"]))


def _region_norm(text: str, size: tuple[int, int]):
    try:
        x0, y0, x1, y1 = (float(v) for v in ast.literal_eval((text or "").strip()))
    except (ValueError, SyntaxError, TypeError):
        return None
    w, h = size
    return [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)]


def to_item(rec: dict, size: tuple[int, int], model_rows: dict) -> dict:
    """What the labelling page needs for one image."""
    annotations = {}
    for rid in RULE_IDS:
        v = rec.get(f"{rid}_violation")
        if v:
            annotations[rid] = {"reason": v.get("reason", "") or "",
                                "boxes": [list(map(float, b)) for b in v.get("bounding_box") or []]}
    model = {}
    for rid, row in model_rows.items():
        model[rid] = {"verdict": row.get("verdict", ""),
                      "region": _region_norm(row.get("region", ""), size),
                      "rationale": row.get("rationale", "")}
    return {
        "image_id": rec["image_id"],
        "quality_of_info": rec.get("quality_of_info", ""),
        "illumination": rec.get("illumination", ""),
        "size": list(size),
        "applicable": {rid: rule_is_applicable(rid, rec) for rid in RULE_IDS},
        "annotations": annotations,
        "model": model,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export", default=None, help="directory for images + items.json")
    args = ap.parse_args()

    spec = __import__("importlib.util").util.spec_from_file_location(
        "s1", STUDY_ROOT / "scripts" / "01_build_sample.py")
    s1 = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(s1)
    from xai_pilot.data import load_construction_site

    print("scanning metadata ...")
    records = s1.scan_records(load_construction_site)
    chosen = draw(records, seed=args.seed)
    out = config.RESULTS_DIR / "audit_manifest.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "quality_of_info", "violated_rules"])
        for r in chosen:
            w.writerow([r["image_id"], r["quality_of_info"],
                        "|".join(rid for rid in RULE_IDS if r.get(f"{rid}_violation"))])
    n_h = sum(any(r.get(f"{rid}_violation") for rid in RULE_IDS) for r in chosen)
    print(f"wrote {out}: {len(chosen)} images, {n_h} with an annotated violation")
    if not args.export:
        return 0

    baseline = {}
    bpath = config.RESULTS_DIR / "baseline_qwen7b-4bit.csv"
    if bpath.exists():
        with bpath.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                baseline.setdefault(row["image_id"], {})[row["rule_id"]] = row

    exp = Path(args.export)
    (exp / "img").mkdir(parents=True, exist_ok=True)
    by_id = {r["image_id"]: r for r in chosen}
    items = {}
    for split in config.SPLITS:
        for rec in load_construction_site(split=split, streaming=True):
            if rec["image_id"] not in by_id or rec["image_id"] in items:
                continue
            img = rec["image"].convert("RGB")
            items[rec["image_id"]] = to_item(by_id[rec["image_id"]], img.size,
                                             baseline.get(rec["image_id"], {}))
            img.thumbnail((LONG_SIDE, LONG_SIDE))
            img.save(exp / "img" / f"{rec['image_id']}.jpg", quality=82)
            if len(items) == len(by_id):
                break
        if len(items) == len(by_id):
            break
    ordered = [items[r["image_id"]] for r in chosen if r["image_id"] in items]
    (exp / "items.json").write_text(json.dumps(ordered), encoding="utf-8")
    print(f"exported {len(ordered)} images to {exp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
