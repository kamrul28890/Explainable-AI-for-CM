"""Day 5 deliverable: descriptive accuracy over every Day-3 sample.

Reuses Day 3's boxes (results/baseline_predictions.csv) to rank regions the
same way Day 4 did, then actually reruns Florence-2 with the top-1 (and
top-1+top-2) region masked out, to see whether the proxy's answer flips.
"""

import json
import sys

import pandas as pd

from xai_pilot.config import FIGURES_DIR, RESULTS_DIR
from xai_pilot.data import load_construction_site
from xai_pilot.inference import AnswerResult
from xai_pilot.metrics.descriptive_accuracy import evaluate
from xai_pilot.model import load_florence2
from xai_pilot.regions import standardize_regions
from xai_pilot.viz import overlay_boxes, save_figure

N_VISUALIZE = 10
OBJECT_LABEL = {"rule_1": "hard hat", "rule_2": "harness", "rule_3": "guardrail", "rule_4": "excavator"}


def main() -> int:
    preds_df = pd.read_csv(RESULTS_DIR / "baseline_predictions.csv", dtype=str)
    target_ids = set(preds_df["image_id"])

    print("Loading Florence-2-base-ft...")
    model, processor = load_florence2()

    print(f"Streaming test split to fetch {len(target_ids)} images...")
    ds = load_construction_site(split="test", streaming=True)
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
    for i, row in preds_df.iterrows():
        image_id = row["image_id"]
        rule_id = row["assigned_rule_id"]
        image = images_by_id[image_id]

        worker_boxes = json.loads(row["worker_boxes"])
        object_boxes = json.loads(row["object_boxes"])
        boxes = worker_boxes + object_boxes
        labels = ["worker"] * len(worker_boxes) + [OBJECT_LABEL[rule_id]] * len(object_boxes)
        regions = standardize_regions(boxes, labels, image_size=(image.width, image.height))

        baseline = AnswerResult(
            answer=row["answer"],
            boxes=boxes,
            confidence=float(row["confidence"]),
            worker_boxes=worker_boxes,
            object_boxes=object_boxes,
        )

        result = evaluate(model, processor, image, rule_id, baseline, regions)

        out_rows.append(
            {
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
        )

        if visualized < N_VISUALIZE and result.answer_changed_top1:
            top1_overlay = overlay_boxes(image, [regions[0].box], labels=["masked top-1"], color="lime")
            save_figure(top1_overlay, fig_dir / f"{image_id}_{rule_id}_flip.png")
            visualized += 1

        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(preds_df)} done...")

    out_csv = RESULTS_DIR / "descriptive_accuracy.csv"
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {out_csv}")
    print(f"Saved {visualized} answer-flip overlay images to {fig_dir}")

    print()
    print(f"Overall top-1 descriptive accuracy: {out_df['answer_changed_top1'].mean():.1%}")
    print(f"Overall top-2 descriptive accuracy: {out_df['answer_changed_top2'].mean():.1%}")
    print()
    print("By primary_class (top-1):")
    print(out_df.groupby("primary_class")["answer_changed_top1"].mean())
    print()
    print("By assigned_rule_id (top-1):")
    print(out_df.groupby("assigned_rule_id")["answer_changed_top1"].mean())
    return 0


if __name__ == "__main__":
    sys.exit(main())
