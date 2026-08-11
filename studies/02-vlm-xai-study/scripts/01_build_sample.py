"""Stage 1 -- build the two-stratum sample manifest.

Runs on any machine: it reads dataset metadata only, no model involved.

Writes results/sample_manifest.csv, which defines the sample for every later stage.
FREEZE IT once written. Regenerating it mid-study silently invalidates everything
already computed, because stage 4's rows would refer to different images than
stage 2's.

    python scripts/01_build_sample.py --seed 42
"""
import argparse
import sys

from xai_vlm.sampling import summarize_strata


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/sample_manifest.csv")
    ap.parse_args()

    print(__doc__)
    print("NOT YET IMPLEMENTED.\n")
    print("To implement, this stage must:")
    print("  1. scan all 10,013 records for violations, scene metadata, quality_of_info")
    print("  2. take all 1,278 violation images            -> stratum H")
    print("  3. match 1,278 compliant images on metadata   -> stratum C")
    print("     (xai_vlm.sampling.match_compliant_sample)")
    print("  4. choose rules per image (xai_vlm.rules.rules_to_ask)")
    print("  5. print xai_vlm.sampling.stratification_report and check for WARNING")
    print("  6. write the manifest, then commit it")
    print("\nHelpers are implemented and tested:")
    print("  match_compliant_sample, stratification_report, summarize_strata,")
    print("  reweight_precision  (see tests/test_rules_and_sampling.py)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
