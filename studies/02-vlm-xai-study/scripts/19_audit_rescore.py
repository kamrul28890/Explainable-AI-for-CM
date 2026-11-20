"""Section 5.5 -- score both tiers against the adjudicated audit labels.

Every Study 2 accuracy figure is scored against the dataset annotation. The
section 5.5 audit relabelled 100 images (first reader blind, owner adjudicated);
61 of them are in the study sample and carry a model answer. This scores each
tier on those questions twice, against the annotation and against the
adjudicated labels, on the same question set: a rule the adjudicated reader
could not tell is dropped from both. It then lists every question where the two
references disagree and which one the model sided with.

The draw is stratified 25 hazard + 25 compliant images per quality tier and the
sample asks only some rules per image, so these rates are as drawn, not
corpus estimates. They answer one question: does correcting the reference move
the model's figures, and in which direction.

    python scripts/19_audit_rescore.py > results/audit_rescore.txt
"""
import argparse
import csv
import json
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402
from xai_vlm.metrics.reporting import rate_row  # noqa: E402

TIERS = ("qwen7b-4bit", "qwen7b-bf16")
csv.field_size_limit(10 ** 8)


def pairs(items: list[dict], answers: dict, rows: list[dict]) -> list[dict]:
    """One entry per (image, applicable rule) that the model answered in the sample."""
    by_id = {it["image_id"]: it for it in items}
    out = []
    for r in rows:
        it = by_id.get(r["image_id"])
        if it is None or not (it.get("applicable") or {}).get(r["rule_id"], False):
            continue
        a = ((answers.get(r["image_id"]) or {}).get("blind") or {}).get(r["rule_id"], "")
        out.append({"image_id": r["image_id"], "rule_id": r["rule_id"],
                    "tier": it.get("quality_of_info", ""),
                    "annotation": r["rule_id"] in (it.get("annotations") or []),
                    "audit": {"yes": True, "no": False}.get(a),
                    "model": str(r["verdict"]).upper() == "YES"})
    return out


def score(ps: list[dict], ref: str) -> dict:
    """Confusion counts against one reference, over questions the audit answered."""
    ps = [p for p in ps if p["audit"] is not None]
    c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": len(ps)}
    for p in ps:
        c[("t" if p["model"] == p[ref] else "f") + ("p" if p["model"] else "n")] += 1
    c["pos"] = [p["model"] for p in ps if p[ref]]
    c["neg"] = [not p["model"] for p in ps if not p[ref]]
    c["acc"] = [p["model"] == p[ref] for p in ps]
    return c


def flips(ps: list[dict]) -> list[dict]:
    """Questions where the audit and the annotation disagree, and whom the model sides with."""
    return [{"image_id": p["image_id"], "rule_id": p["rule_id"], "annotation": p["annotation"],
             "audit": p["audit"], "model": p["model"],
             "sides_with": "audit" if p["model"] == p["audit"] else "annotation"}
            for p in ps if p["audit"] is not None and p["audit"] != p["annotation"]]


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", default=str(STUDY_ROOT / "audit" / "items.json"))
    ap.add_argument("--answers", default=str(config.RESULTS_DIR / "audit_answers_adjudicated"
                                             / "audit" / "owner-adjudicated" / "images"))
    ap.add_argument("--baseline-dir", default=str(config.RESULTS_DIR / "scholar-a40"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args(argv)

    items = json.loads(Path(a.items).read_text(encoding="utf-8"))
    answers = {}
    for f in sorted(Path(a.answers).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        answers[d.get("image_id") or f.stem] = d

    for tier in TIERS:
        ps = pairs(items, answers, _read(Path(a.baseline_dir) / f"baseline_{tier}.csv"))
        imgs = {p["image_id"] for p in ps}
        unsure = sum(p["audit"] is None for p in ps)
        print(f"== {tier}: {len(imgs)} audit images in the sample, {len(ps)} questions, "
              f"{unsure} dropped as can't tell")
        print(f"  {'':<13} {'against annotation':<30} against adjudicated labels")
        sa, sb = score(ps, "annotation"), score(ps, "audit")
        for label, key in (("sensitivity", "pos"), ("specificity", "neg"), ("accuracy", "acc")):
            ra, rb = rate_row(label, sa[key], a.n_boot), rate_row(label, sb[key], a.n_boot)
            print(f"  {label:<13} {ra['formatted']:<30} {rb['formatted']}")
        print(f"  violations    {sum(p['annotation'] for p in ps if p['audit'] is not None):<30} "
              f"{sum(bool(p['audit']) for p in ps)}")
        fl = flips(ps)
        side = sum(f["sides_with"] == "audit" for f in fl)
        print(f"  references disagree on {len(fl)} questions; the model sides with the "
              f"adjudicated label on {side}")
        for f in fl:
            print(f"    {f['image_id']} {f['rule_id']}  annotation {'Y' if f['annotation'] else 'N'}"
                  f"  audit {'Y' if f['audit'] else 'N'}  model {'Y' if f['model'] else 'N'}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
