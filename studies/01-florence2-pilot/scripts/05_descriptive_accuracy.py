"""Day 5 deliverable: descriptive accuracy over every Day-3 sample.

Reuses Day 3's boxes (results/baseline_predictions.csv) to rank regions the
same way Day 4 did, then actually reruns Florence-2 with the top-1 (and
top-1+top-2) region masked out, to see whether the proxy's answer flips.
"""

import argparse
import json
import sys

import pandas as pd

from xai_pilot import config
from xai_pilot.config import (
    FIGURES_DIR,
    REGION_RANKING,
    REPORT_WORKER_LOSS_CORRECTED,
    RESULTS_DIR,
    TEXT_ABLATION_PHRASE,
)
from xai_pilot.data import load_construction_site
from xai_pilot.inference import AnswerResult, answer_rule_text_ablated, graded_score_for
from xai_pilot.metrics.descriptive_accuracy import evaluate
from xai_pilot.model import load_florence2
from xai_pilot.prompts import RULE_OBJECT_LABEL, rule_object_labels
from xai_pilot.regions import REGION_RANKING_MODES, standardize_regions
from xai_pilot.viz import overlay_boxes, save_figure

N_VISUALIZE = 10
OBJECT_LABEL = RULE_OBJECT_LABEL


def main(
    region_ranking: str = REGION_RANKING,
    report_worker_loss_corrected: bool = REPORT_WORKER_LOSS_CORRECTED,
    decoding: str = None,
    mask_mode: str = "black",
    report_graded: bool = False,
    report_text_ablation: bool = False,
) -> int:
    """Measure answer changes after masking the top one and two regions.

    `region_ranking` defaults to config.REGION_RANKING ("area", frozen pilot).
    "rule_aware" makes the top-1 masked region the rule's queried object
    (e.g. the hard hat) instead of the worker body; output is written to a
    mode-suffixed CSV so the frozen record is preserved.

    `report_worker_loss_corrected` (Phase 1.4) additionally logs post-mask
    worker boxes and reports a genuine (worker-loss-excluded) flip rate beside
    the raw one; it also suffixes the output file ("_wlc").

    `decoding` (Phase 1.6) sets the global grounding decoding policy ("beam" or
    "greedy"); non-default adds a "greedy" suffix to the output file.
    """
    if decoding is not None:
        config.DECODING = decoding
    decoding = config.DECODING
    preds_df = pd.read_csv(RESULTS_DIR / "baseline_predictions.csv", dtype=str)
    target_ids = set(preds_df["image_id"])

    print("Loading Florence-2-base-ft...")
    model, processor = load_florence2()

    print(f"Streaming test split to fetch {len(target_ids)} images...")
    ds = load_construction_site(split="test", streaming=True)
    # Keep images in memory because each one is used for multiple masked
    # re-inference calls during this script.
    images_by_id = {}
    for row in ds:
        if row["image_id"] in target_ids:
            images_by_id[row["image_id"]] = row["image"]
        if len(images_by_id) >= len(target_ids):
            break

    fig_dir = FIGURES_DIR / "descriptive_accuracy"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    visualized = 0
    # Rebuild the exact baseline objects and region order from saved Day 3
    # outputs so the unmasked model does not need to run again.
    for i, row in preds_df.iterrows():
        image_id = row["image_id"]
        rule_id = row["assigned_rule_id"]
        image = images_by_id[image_id]

        worker_boxes = json.loads(row["worker_boxes"])
        object_boxes = json.loads(row["object_boxes"])
        boxes = worker_boxes + object_boxes
        labels = ["worker"] * len(worker_boxes) + [OBJECT_LABEL[rule_id]] * len(object_boxes)
        regions = standardize_regions(
            boxes,
            labels,
            image_size=(image.width, image.height),
            region_ranking=region_ranking,
            object_labels=rule_object_labels(rule_id),
        )

        baseline = AnswerResult(
            answer=row["answer"],
            boxes=boxes,
            confidence=float(row["confidence"]),
            worker_boxes=worker_boxes,
            object_boxes=object_boxes,
            # Recompute the continuous baseline score from the saved boxes so
            # graded_score_drop is well-defined without re-running the baseline.
            graded_score=graded_score_for(rule_id, worker_boxes, object_boxes, (image.width, image.height)),
        )

        # `evaluate` applies cumulative masks: first top-1, then top-1 plus
        # top-2, with a fresh safety-proxy inference after each condition.
        result = evaluate(model, processor, image, rule_id, baseline, regions, mask_mode=mask_mode)

        record = {
            "image_id": image_id,
            "primary_class": row["primary_class"],
            "assigned_rule_id": rule_id,
            "baseline_answer": baseline.answer,
            "n_regions": len(regions),
            "answer_changed_top1": result.answer_changed_top1,
            "answer_changed_top2": result.answer_changed_top2,
            "masked_answer_top1": result.masked_answer_top1,
            "masked_answer_top2": result.masked_answer_top2,
            "confidence_drop_top1": result.confidence_drop_top1,
            "confidence_drop_top2": result.confidence_drop_top2,
        }
        if report_worker_loss_corrected:
            record.update(
                {
                    "worker_lost_top1": result.worker_lost_top1,
                    "worker_lost_top2": result.worker_lost_top2,
                    "flip_due_to_worker_loss_top1": result.flip_due_to_worker_loss_top1,
                    "flip_due_to_worker_loss_top2": result.flip_due_to_worker_loss_top2,
                }
            )
        if report_graded:
            record.update(
                {
                    "baseline_graded_score": baseline.graded_score,
                    "graded_score_drop_top1": result.graded_score_drop_top1,
                    "graded_score_drop_top2": result.graded_score_drop_top2,
                    "non_monotonic": result.non_monotonic,
                }
            )
        if report_text_ablation:
            # PROXY APPROXIMATION of native-VQA text-token ablation: reuse the
            # baseline worker detection, swap the specific object phrase for a
            # concept-free placeholder, and re-decide. Reported separately.
            text_ablated_answer, _ = answer_rule_text_ablated(
                model, processor, image, rule_id, worker_boxes, TEXT_ABLATION_PHRASE
            )
            record.update(
                {
                    "text_ablated_answer": text_ablated_answer,
                    "text_ablation_flip": text_ablated_answer != baseline.answer,
                }
            )
        out_rows.append(record)

        # Prefer visually informative answer-flip cases over arbitrary samples.
        if visualized < N_VISUALIZE and result.answer_changed_top1:
            top1_overlay = overlay_boxes(image, [regions[0].box], labels=["masked top-1"], color="lime")
            save_figure(top1_overlay, fig_dir / f"{image_id}_{rule_id}_flip.png")
            visualized += 1

        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(preds_df)} done...")

    parts = [p for p in (
        region_ranking if region_ranking != "area" else "",
        "greedy" if decoding != "beam" else "",
        mask_mode if mask_mode != "black" else "",
        "wlc" if report_worker_loss_corrected else "",
        "graded" if report_graded else "",
        "textabl" if report_text_ablation else "",
    ) if p]
    suffix = ("_" + "_".join(parts)) if parts else ""
    out_csv = RESULTS_DIR / f"descriptive_accuracy{suffix}.csv"
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Ranking policy: {region_ranking} | mask mode: {mask_mode}")
    print(f"Wrote {len(out_df)} rows to {out_csv}")
    print(f"Saved {visualized} answer-flip overlay images to {fig_dir}")

    print()
    print(f"Overall top-1 descriptive accuracy (raw): {out_df['answer_changed_top1'].mean():.1%}")
    print(f"Overall top-2 descriptive accuracy (raw): {out_df['answer_changed_top2'].mean():.1%}")
    if report_graded:
        gd1 = out_df["graded_score_drop_top1"].dropna()
        n_nonmono = int(out_df["non_monotonic"].sum())
        print()
        print(f"Graded-score drop top-1 (mean over {len(gd1)} scored): {gd1.mean():.3f}")
        print(f"Samples with a non-zero graded drop but NO answer flip: "
              f"{int(((out_df['graded_score_drop_top1'].fillna(0) > 0) & ~out_df['answer_changed_top1']).sum())}")
        print(f"Non-monotonic (top-2 un-flips top-1) samples: {n_nonmono}")
    if report_text_ablation:
        tflip = out_df["text_ablation_flip"].mean()
        print()
        print("TEXT-ABLATION (proxy approximation -- Florence-2 is a grounding proxy, NOT native-VQA;")
        print(f"  swapped the specific object phrase for '{TEXT_ABLATION_PHRASE}'. Report separately.):")
        print(f"  text-ablation answer-flip rate: {tflip:.1%}")
        print(f"  (compare to visual-ablation top-1 flip rate: {out_df['answer_changed_top1'].mean():.1%})")
    if report_worker_loss_corrected:
        # Genuine flip = answer changed AND not attributable to worker loss.
        genuine_top1 = out_df["answer_changed_top1"] & ~out_df["flip_due_to_worker_loss_top1"]
        genuine_top2 = out_df["answer_changed_top2"] & ~out_df["flip_due_to_worker_loss_top2"]
        print()
        print(f"Worker-loss flips top-1: {int(out_df['flip_due_to_worker_loss_top1'].sum())} "
              f"of {int(out_df['answer_changed_top1'].sum())} flips")
        print(f"Overall top-1 descriptive accuracy (genuine): {genuine_top1.mean():.1%}")
        print(f"Overall top-2 descriptive accuracy (genuine): {genuine_top2.mean():.1%}")
    print()
    print("By primary_class (top-1, raw):")
    print(out_df.groupby("primary_class")["answer_changed_top1"].mean())
    print()
    print("By assigned_rule_id (top-1, raw):")
    print(out_df.groupby("assigned_rule_id")["answer_changed_top1"].mean())
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region-ranking",
        choices=REGION_RANKING_MODES,
        default=REGION_RANKING,
        help="Candidate-region ranking policy (default: config.REGION_RANKING).",
    )
    parser.add_argument(
        "--report-worker-loss-corrected",
        action="store_true",
        default=REPORT_WORKER_LOSS_CORRECTED,
        help="Log worker-loss flags and report genuine (worker-loss-excluded) flip rate.",
    )
    parser.add_argument(
        "--decoding",
        choices=["beam", "greedy"],
        default=config.DECODING,
        help="Grounding decoding policy (default: config.DECODING).",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["black", "blur", "inpaint"],
        default="black",
        help="Masking style for the region ablation (default: black, frozen).",
    )
    parser.add_argument(
        "--report-graded",
        action="store_true",
        default=False,
        help="Log graded-score drop + non-monotonic flag and report their stats.",
    )
    parser.add_argument(
        "--report-text-ablation",
        action="store_true",
        default=False,
        help="Add the proxy text-ablation probe (reported separately from native-VQA).",
    )
    args = parser.parse_args()
    sys.exit(main(
        region_ranking=args.region_ranking,
        report_worker_loss_corrected=args.report_worker_loss_corrected,
        decoding=args.decoding,
        mask_mode=args.mask_mode,
        report_graded=args.report_graded,
        report_text_ablation=args.report_text_ablation,
    ))
