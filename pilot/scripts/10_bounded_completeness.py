"""Day 10 deliverable: bucket each sample into the bounded-completeness verdict.

The headline classification is a pure rollup of Day 4's region_extraction.csv
(top_region_source) and Day 5's descriptive_accuracy.csv (answer_changed_top1/2)
-- no new model calls. Day 5 didn't log post-mask worker_boxes though, so this
script makes one small, explicitly bounded set of fresh top-1-mask reruns
(only for samples with a real model region -- 154 of 163) specifically to
check the plan's flagged risk: does masking the top region sometimes remove
the only detected worker and trigger _answer_presence_rule's scene-level
fallback branch, the same artifact Day 5 found and Day 9 quantified for
perturbations? That would make a sample look "explanation_supported" for a
reason disconnected from the region actually containing the safety object.
"""

import sys

import pandas as pd

from xai_pilot.config import DATA_DIR, RESULTS_DIR
from xai_pilot.data import load_construction_site
from xai_pilot.inference import AnswerResult, answer_rule
from xai_pilot.metrics.completeness import classify_sample
from xai_pilot.metrics.robustness import _robustness_from_results
from xai_pilot.regions import mask_region

import json


def _baseline_from_row(row) -> AnswerResult:
    """Rehydrate the saved baseline fields needed for fallback-risk analysis."""
    return AnswerResult(
        answer=row["answer"],
        worker_boxes=[tuple(b) for b in json.loads(row["worker_boxes"])],
        object_boxes=[tuple(b) for b in json.loads(row["object_boxes"])],
        confidence=float(row["confidence"]),
    )


def main() -> int:
    """Assign completeness verdicts and audit worker-loss artifacts."""
    # Read booleans as strings first, then map explicitly. Python's bool("False")
    # is True, so a direct astype(bool) would corrupt these columns.
    accuracy_df = pd.read_csv(RESULTS_DIR / "descriptive_accuracy.csv", dtype=str)
    accuracy_df["answer_changed_top1"] = accuracy_df["answer_changed_top1"].map({"True": True, "False": False})
    accuracy_df["answer_changed_top2"] = accuracy_df["answer_changed_top2"].map({"True": True, "False": False})

    regions_df = pd.read_csv(RESULTS_DIR / "region_extraction.csv", dtype=str)
    samples_df = pd.read_csv(DATA_DIR / "pilot_samples.csv", dtype=str)
    preds_df = pd.read_csv(RESULTS_DIR / "baseline_predictions.csv", dtype=str)

    # Compose one analysis table from prior-day artifacts. Each merge adds only
    # the fields required for verdicts or the hard-subset diagnostics.
    merged = accuracy_df.merge(
        regions_df[["image_id", "assigned_rule_id", "top_region_source", "top_region_box"]],
        on=["image_id", "assigned_rule_id"],
    )
    merged = merged.merge(samples_df[["image_id", "violated_rule_ids"]], on="image_id")
    merged = merged.merge(preds_df[["image_id", "answer", "worker_boxes", "object_boxes", "confidence"]], on="image_id")
    merged["is_multi_rule_violation"] = merged["violated_rule_ids"].apply(
        lambda s: isinstance(s, str) and "|" in s
    )

    target_ids = set(merged["image_id"])
    print(f"Streaming test split to fetch quality_of_info for {len(target_ids)} images...")
    ds = load_construction_site(split="test", streaming=True)
    # Fetch metadata and image pixels in separate streaming passes. The first
    # covers all samples; the second downloads images only where a real model
    # region can be masked for the fallback audit.
    quality_by_id = {}
    for row in ds:
        if row["image_id"] in target_ids:
            quality_by_id[row["image_id"]] = row["quality_of_info"]
        if len(quality_by_id) >= len(target_ids):
            break
    images_by_id = {}
    ds2 = load_construction_site(split="test", streaming=True)
    model_region_ids = set(merged[merged["top_region_source"] == "model"]["image_id"])
    for row in ds2:
        if row["image_id"] in model_region_ids:
            images_by_id[row["image_id"]] = row["image"]
        if len(images_by_id) >= len(model_region_ids):
            break

    print("Loading Florence-2-base-ft for the top-1-mask fallback-risk check...")
    from xai_pilot.model import load_florence2

    model, processor = load_florence2()

    out_rows = []
    for i, row in merged.iterrows():
        verdict = classify_sample(row["top_region_source"], row["answer_changed_top1"], row["answer_changed_top2"])

        worker_lost_in_top1_mask = None
        # Grid fallbacks have no model-selected region and are already labeled
        # no_usable_explanation, so rerunning them would not answer the audit
        # question and would waste GPU work.
        if row["top_region_source"] == "model":
            baseline = _baseline_from_row(row)
            top1_box = tuple(json.loads(row["top_region_box"]))
            image = images_by_id[row["image_id"]]
            masked = mask_region(image, top1_box, mode="black")
            masked_result = answer_rule(model, processor, masked, row["assigned_rule_id"])
            worker_lost_in_top1_mask = _robustness_from_results(baseline, masked_result).worker_lost

        out_rows.append(
            {
                "image_id": row["image_id"],
                "assigned_rule_id": row["assigned_rule_id"],
                "primary_class": row["primary_class"],
                "verdict": verdict,
                "is_multi_rule_violation": row["is_multi_rule_violation"],
                "quality_of_info": quality_by_id.get(row["image_id"]),
                "top_region_source": row["top_region_source"],
                "worker_lost_in_top1_mask": worker_lost_in_top1_mask,
            }
        )
        if (i + 1) % 40 == 0:
            print(f"{i + 1}/{len(merged)} done...")

    out_df = pd.DataFrame(out_rows)
    out_csv = RESULTS_DIR / "bounded_completeness.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_csv}")

    print("\nOverall verdict distribution:")
    print(out_df["verdict"].value_counts(normalize=True))

    print("\nVerdict distribution for quality_of_info == 'poor info' (hard subset):")
    poor_info = out_df[out_df["quality_of_info"] == "poor info"]
    print(f"n={len(poor_info)}")
    if len(poor_info):
        print(poor_info["verdict"].value_counts(normalize=True))

    print("\nVerdict distribution for multi-rule-violation samples (hard subset):")
    multi = out_df[out_df["is_multi_rule_violation"]]
    print(f"n={len(multi)}")
    if len(multi):
        print(multi["verdict"].value_counts(normalize=True))

    print("\nGrid-fallback rate (no_usable_explanation by definition):")
    print(f"{(out_df['top_region_source'] == 'grid').mean():.1%}")

    # Only model-region samples received the fresh mask rerun. Restrict the
    # worker-loss calculation to that explicitly observed subset.
    scored = out_df[out_df["worker_lost_in_top1_mask"].notna()].copy()
    scored["worker_lost_in_top1_mask"] = scored["worker_lost_in_top1_mask"].astype(bool)
    print(f"\nworker_lost_in_top1_mask rate over the {len(scored)} model-region samples: {scored['worker_lost_in_top1_mask'].mean():.1%}")
    supported = scored[scored["verdict"] == "explanation_supported"]
    print(
        f"Of the {len(supported)} explanation_supported verdicts, "
        f"{supported['worker_lost_in_top1_mask'].mean():.1%} also show worker_lost_in_top1_mask "
        "(possible fallback-rerouting artifact rather than a genuinely load-bearing region)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
