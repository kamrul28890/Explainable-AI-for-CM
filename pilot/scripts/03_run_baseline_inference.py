"""Day 3 deliverable: real baseline Florence-2 outputs for every pilot sample.

Streams the test split once; for each row whose image_id is in
pilot_samples.csv, runs the assigned safety-rule grounding query
(inference.answer_rule) plus a <MORE_DETAILED_CAPTION> call (logged for
context, not scored), and saves box overlays for a handful of samples for
manual sanity-checking.
"""

import json
import sys

import pandas as pd

from xai_pilot.config import DATA_DIR, FIGURES_DIR, RESULTS_DIR
from xai_pilot.data import load_construction_site
from xai_pilot.inference import answer_rule
from xai_pilot.model import load_florence2, run_task
from xai_pilot.viz import overlay_boxes, save_figure

N_VISUALIZE = 20
CAPTION_TASK = "<MORE_DETAILED_CAPTION>"


def main() -> int:
    """Run and persist baseline predictions for every selected sample."""
    # These lookup tables turn the sequential dataset stream into an efficient
    # membership test and preserve the experimental assignment from Day 2.
    samples_df = pd.read_csv(DATA_DIR / "pilot_samples.csv", dtype=str)
    target_ids = set(samples_df["image_id"])
    rule_by_id = dict(zip(samples_df["image_id"], samples_df["assigned_rule_id"]))
    class_by_id = dict(zip(samples_df["image_id"], samples_df["primary_class"]))

    print("Loading Florence-2-base-ft...")
    model, processor = load_florence2()

    print(f"Streaming test split, looking for {len(target_ids)} target samples...")
    ds = load_construction_site(split="test", streaming=True)

    fig_dir = FIGURES_DIR / "baseline"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # The dataset is streamed once. Non-pilot rows are skipped immediately, so
    # model inference is performed only for the fixed target IDs.
    out_rows = []
    visualized = 0
    for row in ds:
        image_id = row["image_id"]
        if image_id not in target_ids:
            continue

        rule_id = rule_by_id[image_id]
        image = row["image"]

        # The detailed caption is qualitative context for manual review. It is
        # not used to derive the safety answer or any reported metric.
        _, caption_parsed, _ = run_task(model, processor, image, CAPTION_TASK)
        caption = caption_parsed.get(CAPTION_TASK, "")

        result = answer_rule(model, processor, image, rule_id)

        # JSON preserves a variable number of boxes inside a flat CSV row.
        out_rows.append(
            {
                "image_id": image_id,
                "primary_class": class_by_id[image_id],
                "assigned_rule_id": rule_id,
                "answer": result.answer,
                "worker_boxes": json.dumps(result.worker_boxes),
                "object_boxes": json.dumps(result.object_boxes),
                "confidence": result.confidence,
                "inference_ms": result.inference_ms,
                "caption": caption,
            }
        )

        # Save a bounded number of overlays to inspect grounding quality without
        # generating hundreds of redundant image files.
        if visualized < N_VISUALIZE and result.boxes:
            overlaid = overlay_boxes(
                image, result.worker_boxes, labels=["worker"] * len(result.worker_boxes), color="blue"
            )
            overlaid = overlay_boxes(
                overlaid, result.object_boxes, labels=[rule_id] * len(result.object_boxes), color="red"
            )
            save_figure(overlaid, fig_dir / f"{image_id}_{rule_id}.png")
            visualized += 1

        if len(out_rows) % 20 == 0:
            print(f"{len(out_rows)}/{len(target_ids)} done...")

        if len(out_rows) >= len(target_ids):
            break

    out_csv = RESULTS_DIR / "baseline_predictions.csv"
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(out_rows)} rows to {out_csv}")
    print(f"Saved {visualized} overlay images to {fig_dir}")

    # A non-empty set means the source split and Day 2 manifest disagree. Keep
    # the partial output for diagnosis but report the integrity problem.
    missing = target_ids - {r["image_id"] for r in out_rows}
    if missing:
        print(f"WARNING: {len(missing)} target image_ids not found in test split stream: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
