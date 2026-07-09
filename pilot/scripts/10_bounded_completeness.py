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

import argparse
import sys

import pandas as pd

from xai_pilot.config import DATA_DIR, REGION_RANKING, REPORT_WORKER_LOSS_CORRECTED, RESULTS_DIR
from xai_pilot.data import load_construction_site
from xai_pilot.inference import AnswerResult, answer_rule
from xai_pilot.metrics.completeness import classify_sample, classify_sample_worker_loss_corrected
from xai_pilot.metrics.robustness import _robustness_from_results
from xai_pilot.regions import REGION_RANKING_MODES, mask_region

import json


def _bool_col(series):
    """Map a string boolean column to real bools (bool('False') is True)."""
    return series.map({"True": True, "False": False})


def _rollup_worker_loss_corrected(region_ranking: str) -> int:
    """Bounded-completeness raw vs worker-loss-corrected verdicts, no reruns.

    Phase 1.4 makes descriptive_accuracy log post-mask worker boxes and a
    flip_due_to_worker_loss flag, so the fallback-artifact correction Day 10
    previously needed 159 fresh GPU reruns to estimate is now a pure rollup of
    the "_wlc" descriptive CSV -- no model load, no re-inference.
    """
    ranking_suffix = "" if region_ranking == "area" else f"_{region_ranking}"
    da_name = f"descriptive_accuracy{ranking_suffix}_wlc.csv"
    re_name = f"region_extraction{ranking_suffix}.csv"
    da = pd.read_csv(RESULTS_DIR / da_name, dtype=str)
    for c in ["answer_changed_top1", "answer_changed_top2",
              "flip_due_to_worker_loss_top1", "flip_due_to_worker_loss_top2"]:
        da[c] = _bool_col(da[c])
    re = pd.read_csv(RESULTS_DIR / re_name, dtype=str)
    merged = da.merge(re[["image_id", "assigned_rule_id", "top_region_source"]],
                      on=["image_id", "assigned_rule_id"])

    merged["verdict_raw"] = merged.apply(
        lambda r: classify_sample(r["top_region_source"], r["answer_changed_top1"], r["answer_changed_top2"]),
        axis=1,
    )
    merged["verdict_corrected"] = merged.apply(
        lambda r: classify_sample_worker_loss_corrected(
            r["top_region_source"], r["answer_changed_top1"], r["answer_changed_top2"],
            r["flip_due_to_worker_loss_top1"], r["flip_due_to_worker_loss_top2"],
        ),
        axis=1,
    )
    out_csv = RESULTS_DIR / f"bounded_completeness{ranking_suffix}_wlc.csv"
    merged[["image_id", "assigned_rule_id", "primary_class", "verdict_raw", "verdict_corrected"]].to_csv(
        out_csv, index=False
    )
    print(f"Read {da_name} + {re_name}; wrote {len(merged)} rows to {out_csv} (no model reruns).")
    n = len(merged)
    raw_sup = (merged["verdict_raw"] == "explanation_supported").sum()
    cor_sup = (merged["verdict_corrected"] == "explanation_supported").sum()
    downgraded = ((merged["verdict_raw"] == "explanation_supported") &
                  (merged["verdict_corrected"] != "explanation_supported")).sum()
    print(f"\nexplanation_supported  raw: {raw_sup}/{n} ({raw_sup/n:.1%})   "
          f"corrected: {cor_sup}/{n} ({cor_sup/n:.1%})")
    print(f"Verdicts downgraded from supported by worker-loss correction: {downgraded}")
    return 0


def _baseline_from_row(row) -> AnswerResult:
    """Rehydrate the saved baseline fields needed for fallback-risk analysis."""
    return AnswerResult(
        answer=row["answer"],
        worker_boxes=[tuple(b) for b in json.loads(row["worker_boxes"])],
        object_boxes=[tuple(b) for b in json.loads(row["object_boxes"])],
        confidence=float(row["confidence"]),
    )


def main(
    report_worker_loss_corrected: bool = REPORT_WORKER_LOSS_CORRECTED,
    region_ranking: str = REGION_RANKING,
) -> int:
    """Assign completeness verdicts and audit worker-loss artifacts.

    With `report_worker_loss_corrected` (Phase 1.4), the verdict is a pure
    rollup of the "_wlc" descriptive CSV that already carries post-mask worker
    boxes -- no model reruns -- reporting raw and worker-loss-corrected
    supported rates. The frozen default path is unchanged (and still performs
    the bounded 159-rerun audit that Phase 1.4 supersedes).
    """
    if report_worker_loss_corrected:
        return _rollup_worker_loss_corrected(region_ranking)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-worker-loss-corrected",
        action="store_true",
        default=REPORT_WORKER_LOSS_CORRECTED,
        help="Roll up raw + worker-loss-corrected verdicts from the _wlc descriptive CSV (no reruns).",
    )
    parser.add_argument(
        "--region-ranking",
        choices=REGION_RANKING_MODES,
        default=REGION_RANKING,
        help="Which ranking's descriptive/region CSVs to roll up (default: config.REGION_RANKING).",
    )
    args = parser.parse_args()
    sys.exit(main(
        report_worker_loss_corrected=args.report_worker_loss_corrected,
        region_ranking=args.region_ranking,
    ))
