"""Rule_4 audit -- are the model's excavator "false alarms" real hazards?

The section 5.5 audit found rule_4 (worker near an operating excavator) is the
rule the annotation most often leaves unmarked. Every rule_4 "false alarm" is
scored against that incomplete reference, so this audit relabels a stratified
draw of the study sample's rule_4 questions and corrects both tiers' rule_4
confusion matrix.

The 323 rule_4 rows in the frozen sample are split into cells by stratum
(compliant image C, hazard image with rule_4 unannotated H, rule_4 annotated P)
and by what the two tiers answered (4-bit, bf16; N/Y). Each cell is drawn with
its own seeded generator; small cells are taken whole. Annotated-positive cells
are in the draw so the reader cannot infer the answer from the draw itself, and
they estimate how often a marked rule_4 violation is wrong.

The reader answers rule_4 only, blind to the cell, the annotation and both
models' answers. Images are shown in a seeded random order.

    python scripts/15_rule4_audit_sample.py                 # rule4_audit_manifest.csv
    python scripts/15_rule4_audit_sample.py --export DIR    # + images and queue.json
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
# Cell -> rows drawn. Decided 2026-11-12, before any image was looked at.
PER_CELL = {"C-NN": 12, "C-NY": 12, "C-Y": 12, "H-NN": 12, "H-NY": 12, "H-YY": 12,
            "P-NN": 8, "P-NY": 8, "P-Y": 8}
LONG_SIDE = 1024
ORDER_SEED = 2026


def cell_of(stratum: str, expected: str, v4: str, vbf: str) -> str:
    """The draw cell for one rule_4 row. Cells too small to split are merged
    on the bf16 answer: C-Y (4-bit YES, 8 rows), P-Y (4-bit YES, 18 rows), and
    H-YY also takes any H row where 4-bit said YES."""
    yn = lambda v: "Y" if str(v).upper() == "YES" else "N"  # noqa: E731
    a, b = yn(v4), yn(vbf)
    if str(expected).upper() == "YES":
        return "P-Y" if a == "Y" else f"P-N{b}"
    s = "C" if stratum == "C" else "H"
    if a == "Y":
        return "C-Y" if s == "C" else "H-YY"
    return f"{s}-N{b}"


def rule4_rows(baselines: dict) -> list[dict]:
    """Join the tiers' rule_4 rows: one per image, with each tier's verdict."""
    by = {}
    for tier, rows in baselines.items():
        for r in rows:
            if r["rule_id"] != "rule_4":
                continue
            d = by.setdefault(r["image_id"], {"image_id": r["image_id"],
                                              "stratum": r["stratum"],
                                              "expected": r["expected"]})
            d[tier] = r["verdict"]
    out = [d for d in by.values() if all(t in d for t in baselines)]
    for d in out:
        d["cell"] = cell_of(d["stratum"], d["expected"], d[TIERS[0]], d[TIERS[1]])
    return sorted(out, key=lambda d: d["image_id"])


def draw(rows: list[dict], per_cell: dict = PER_CELL, seed: int = 42) -> list[dict]:
    """Seeded per cell and independent of row order; a cell smaller than its
    quota is taken whole. Each drawn row carries its cell's size."""
    out = []
    for cell, k in per_cell.items():
        pool = sorted((r for r in rows if r["cell"] == cell), key=lambda r: r["image_id"])
        rng = random.Random(f"rule4-{cell}-{seed}")
        for r in (pool if len(pool) <= k else rng.sample(pool, k)):
            out.append({**r, "cell_size": len(pool)})
    return sorted(out, key=lambda r: r["image_id"])


def blind_order(image_ids: list[str], seed: int = ORDER_SEED) -> list[str]:
    ids = sorted(image_ids)
    random.Random(f"rule4-order-{seed}").shuffle(ids)
    return ids


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export", default=None, help="directory for images + queue.json")
    args = ap.parse_args()

    base = {t: _read(config.RESULTS_DIR / "scholar-a40" / f"baseline_{t}.csv") for t in TIERS}
    rows = rule4_rows(base)
    chosen = draw(rows, seed=args.seed)
    out = config.RESULTS_DIR / "rule4_audit_manifest.csv"
    fields = ["image_id", "cell", "cell_size", "stratum", "expected", *TIERS]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(chosen)
    print(f"wrote {out}: {len(chosen)} of {len(rows)} rule_4 rows")
    for cell in PER_CELL:
        sub = [r for r in chosen if r["cell"] == cell]
        print(f"  {cell:<5} drew {len(sub):>2} of {sub[0]['cell_size'] if sub else 0}")
    if not args.export:
        return 0

    from xai_pilot.data import load_construction_site
    exp = Path(args.export)
    (exp / "img").mkdir(parents=True, exist_ok=True)
    want = {r["image_id"] for r in chosen}
    done = set()
    for split in config.SPLITS:
        for rec in load_construction_site(split=split, streaming=True):
            if rec["image_id"] in want and rec["image_id"] not in done:
                img = rec["image"].convert("RGB")
                img.thumbnail((LONG_SIDE, LONG_SIDE))
                img.save(exp / "img" / f"{rec['image_id']}.jpg", quality=82)
                done.add(rec["image_id"])
    # The queue carries ids only: nothing the reader sees names the cell.
    (exp / "queue.json").write_text(json.dumps(blind_order(sorted(done))), encoding="utf-8")
    print(f"exported {len(done)} images to {exp}")
    return 0 if done == want else 1


if __name__ == "__main__":
    sys.exit(main())
