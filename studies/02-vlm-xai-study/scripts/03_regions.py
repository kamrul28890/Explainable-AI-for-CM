"""Stage 3 -- explanation regions, and the image dimensions metric 7 needs.

    python scripts/03_regions.py --model qwen7b-4bit
"""
import argparse
import csv
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402
from xai_vlm.metrics.regions import REGION_FIELDS, build_region_rows, no_region_rate  # noqa: E402


def read_image_sizes(loader, wanted: set) -> dict:
    """image_id -> (width, height), read from headers rather than full decodes.

    PIL opens lazily: `.size` is available from the header without decoding pixel
    data, so this is I/O-bound and far cheaper than decoding 2,556 photographs.
    """
    sizes = {}
    for split in config.SPLITS:
        for rec in loader(split=split, streaming=True):
            iid = rec["image_id"]
            if iid in wanted and iid not in sizes:
                sizes[iid] = rec["image"].size
            if len(sizes) >= len(wanted):
                return sizes
    return sizes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen7b-4bit")
    args = ap.parse_args()

    baseline_path = STUDY_ROOT / "results" / f"baseline_{args.model}.csv"
    if not baseline_path.exists():
        print(f"no baseline at {baseline_path}. Run Stage 2 first.", file=sys.stderr)
        return 1

    with baseline_path.open(newline="", encoding="utf-8") as fh:
        baseline = list(csv.DictReader(fh))
    print(f"baseline: {len(baseline)} rows")

    from xai_pilot.data import load_construction_site

    wanted = {r["image_id"] for r in baseline}
    print(f"reading dimensions for {len(wanted)} images (headers only) ...")
    sizes = read_image_sizes(load_construction_site, wanted)
    print(f"  got {len(sizes)}")

    rows = build_region_rows(baseline, sizes)
    out = STUDY_ROOT / "results" / f"regions_{args.model}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REGION_FIELDS)
        w.writeheader()
        w.writerows(rows)

    missing_dims = sum(1 for r in rows if not r["image_w"])
    print("\n" + "=" * 60)
    print(f"NO-REGION RATE : {no_region_rate(rows):.1%}")
    print("  the share of cases where the system offered no explanation at all")
    yes = [r for r in rows if r["verdict"] == "YES"]
    if yes:
        print(f"  among YES answers : {no_region_rate(yes):.1%}")
    if missing_dims:
        print(f"WARNING: {missing_dims} rows have no image dimensions; "
              "metric 7 will exclude them.")
    print(f"wrote {out}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
