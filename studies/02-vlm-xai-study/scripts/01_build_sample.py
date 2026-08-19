"""Stage 1 -- build the two-stratum sample manifest.

Runs on any machine: it reads dataset metadata only, no model involved.

Writes results/sample_manifest.csv, which defines the sample for every later stage.
FREEZE IT once written. Regenerating it mid-study silently invalidates everything
already computed, because stage 4's rows would refer to different images than
stage 2's -- and both files would still parse, so nothing downstream would notice.
That is why --force exists and why it is loud.

    python scripts/01_build_sample.py --seed 42
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402
from xai_vlm.rules import expected_answer, rules_to_ask, violated_rules  # noqa: E402
from xai_vlm.sampling import (  # noqa: E402
    match_compliant_sample,
    metadata_key,
    stratification_report,
)

# EXACTLY the columns pre-registered in the architecture document, Stage 1.
#
# No `expected` column: `is_violation` is per RULE, not per image, so ground truth
# is already here -- expected = "YES" if is_violation else "NO". Storing it twice
# would add a field that can disagree with its own source.
MANIFEST_FIELDS = [
    "image_id", "rule_id", "stratum", "is_violation", "is_negative_control",
    "quality_of_info", "illumination", "camera_distance", "view",
]

# The Stage 0 median, used only to project wall clock before any GPU time is spent.
#
# OPTIMISTIC BY ABOUT 3x. Stage 0 timed 39 hand-picked gate images; the measured
# median over the full corpus is 8.84 s, because most corpus images reach the
# MAX_PIXELS cap those smaller gate images did not. The printed projection is a
# floor, not an estimate -- see README, "Running Stage 2".
STAGE0_MEDIAN_S = 2.9

# Which splits the sample is drawn from. Defined in config so Stage 2's image
# source cannot drift from it -- see the note there; they did drift once, and the
# result was a run that exited 0 having covered a third of the manifest.
#
# Both splits, giving the full 10,013-record corpus (architecture doc 5.2, 6.2)
# rather than the 3,004-record test split. Violations per split (arch doc 5.3):
#
#              test   train   total
#   rule_1      324     677    1001
#   rule_3       63     109     172
#   rule_2       25      59      84
#   rule_4       24      46      70
#
# `test` alone would sit rule_2 at n=25 and rule_4 at n=24, below the pilot's
# MIN_N_FLOOR of 30, making two of the four rules unreportable.
SPLITS = config.SPLITS


def scan_records(loader, splits: tuple[str, ...] = SPLITS) -> list[dict]:
    """Read metadata for every record in every split.

    The image column is dropped before iteration where the loader supports it.
    Carrying it would decode all 10,013 JPEGs only to discard them.
    """
    records: list[dict] = []
    for split in splits:
        ds = loader(split=split, streaming=True)
        if hasattr(ds, "remove_columns"):
            ds = ds.remove_columns(["image"])
        for rec in ds:
            records.append({k: v for k, v in rec.items() if k != "image"})
    return records


def _row_for(rec: dict, rule_id: str, is_control: bool, stratum: str) -> dict:
    is_violation = not is_control
    # Two independent code paths must agree on ground truth. rules_to_ask decided
    # `is_control`; expected_answer reads the annotation field directly. If they
    # ever disagree the scan is buggy, and a silent disagreement here would put
    # wrong labels under every metric in the study.
    expected = expected_answer(rule_id, rec)
    if (expected == "YES") != is_violation:
        raise AssertionError(
            f"ground-truth disagreement on {rec['image_id']}/{rule_id}: "
            f"rules_to_ask says is_violation={is_violation}, "
            f"expected_answer says {expected}"
        )
    return {
        "image_id": rec["image_id"],
        "rule_id": rule_id,
        "stratum": stratum,
        "is_violation": is_violation,
        "is_negative_control": is_control,
        "quality_of_info": rec.get("quality_of_info", ""),
        "illumination": rec.get("illumination", ""),
        "camera_distance": rec.get("camera_distance", ""),
        "view": rec.get("view", ""),
    }


def build_rows(records: list[dict], seed: int) -> list[dict]:
    """Split into strata, assign rules, and flatten to one row per image-rule pair."""
    hazard = [r for r in records if violated_rules(r)]
    compliant = [r for r in records if not violated_rules(r)]

    target = Counter(metadata_key(r) for r in hazard)
    matched = match_compliant_sample(compliant, target, n_wanted=len(hazard), seed=seed)

    rows: list[dict] = []
    for stratum, recs in (("H", hazard), ("C", matched)):
        for rec in recs:
            for rule_id, is_control in rules_to_ask(rec, seed=seed):
                rows.append(_row_for(rec, rule_id, is_control, stratum))
    return rows


def write_manifest(rows: list[dict], path: Path, force: bool) -> None:
    """Write the manifest, refusing to clobber an existing one without --force."""
    path = Path(path)
    if path.exists() and not force:
        print(f"\nREFUSING to overwrite {path}.", file=sys.stderr)
        print(
            "This manifest defines the sample for every later stage. Regenerating "
            "it invalidates every result already computed against it -- silently, "
            "because the old result files would still parse.\n"
            "If that is genuinely what you want, pass --force.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(STUDY_ROOT / "results" / "sample_manifest.csv"))
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest (invalidates prior results)")
    ap.add_argument("--allow-imbalance", action="store_true",
                    help="continue despite a stratification WARNING")
    args = ap.parse_args()

    from xai_pilot.data import load_construction_site

    print(f"scanning {' + '.join(SPLITS)} (metadata only) ...")
    records = scan_records(load_construction_site)
    print(f"  {len(records)} records")

    hazard = [r for r in records if violated_rules(r)]
    compliant = [r for r in records if not violated_rules(r)]
    target = Counter(metadata_key(r) for r in hazard)
    matched = match_compliant_sample(compliant, target, n_wanted=len(hazard), seed=args.seed)

    report = stratification_report(hazard, matched)
    print("\n" + report)
    if "WARNING" in report and not args.allow_imbalance:
        print(
            "\nABORTING: the compliant pool could not supply the metadata cells the "
            "hazard stratum needs. Comparing the strata would compare scene "
            "conditions rather than model behaviour. Pass --allow-imbalance to "
            "proceed with this stated as a limitation.",
            file=sys.stderr,
        )
        return 1

    rows = build_rows(records, seed=args.seed)
    write_manifest(rows, Path(args.out), force=args.force)

    from xai_vlm.runner import manifest_sha256

    print(f"\nwrote {len(rows)} image-rule pairs to {args.out}")
    print(f"  strata      : H={sum(r['stratum'] == 'H' for r in rows)}  "
          f"C={sum(r['stratum'] == 'C' for r in rows)}")
    print(f"  positives   : {sum(r['is_violation'] for r in rows)}")
    print(f"  controls    : {sum(r['is_negative_control'] for r in rows)}")
    print(f"  sha256      : {manifest_sha256(Path(args.out))}")
    print(f"\nStage 2 projection at the Stage 0 median of {STAGE0_MEDIAN_S}s/call: "
          f"{len(rows) * STAGE0_MEDIAN_S / 3600:.1f} h")
    print("\nFREEZE THIS FILE. Commit it before running Stage 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
