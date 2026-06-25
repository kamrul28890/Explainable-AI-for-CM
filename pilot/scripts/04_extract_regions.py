"""Day 4 deliverable: standardized explanation regions for every Day-3 sample.

Reuses the worker/object boxes already produced by Day 3's baseline run
(results/baseline_predictions.csv) -- no model re-inference needed here.
For each sample, ranks candidate regions via regions.standardize_regions and
records whether the top region came from the model or the grid fallback.
Saves before/after masked-image pairs for a handful of samples so the
masking itself can be sanity-checked visually before Day 5 relies on it.
"""

import json
import sys

import pandas as pd
from PIL import Image

from xai_pilot.config import DATA_DIR, FIGURES_DIR, RESULTS_DIR
from xai_pilot.data import load_construction_site
from xai_pilot.regions import mask_region, standardize_regions
from xai_pilot.viz import overlay_boxes, save_figure

N_VISUALIZE = 10
OBJECT_LABEL = {"rule_1": "hard hat", "rule_2": "harness", "rule_3": "guardrail", "rule_4": "excavator"}


def main() -> int:
    """Convert baseline grounding boxes into ranked explanation regions."""
    preds_df = pd.read_csv(RESULTS_DIR / "baseline_predictions.csv", dtype=str)
    target_ids = set(preds_df["image_id"])

    print(f"Streaming test split to fetch {len(target_ids)} images for region extraction...")
    ds = load_construction_site(split="test", streaming=True)
    # Day 4 needs the source pixels for masking but does not rerun the model.
    # Cache only selected images while walking the streamed split.
    images_by_id = {}
    for row in ds:
        if row["image_id"] in target_ids:
            images_by_id[row["image_id"]] = row["image"]
        if len(images_by_id) >= len(target_ids):
            break

    fig_dir = FIGURES_DIR / "regions"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    visualized = 0
    # Reconstruct variable-length boxes from JSON, retain semantic labels, and
    # apply the same ranking policy that all later masking metrics consume.
    for _, row in preds_df.iterrows():
        image_id = row["image_id"]
        rule_id = row["assigned_rule_id"]
        image = images_by_id[image_id]

        worker_boxes = json.loads(row["worker_boxes"])
        object_boxes = json.loads(row["object_boxes"])
        boxes = worker_boxes + object_boxes
        labels = ["worker"] * len(worker_boxes) + [OBJECT_LABEL[rule_id]] * len(object_boxes)

        regions = standardize_regions(boxes, labels, image_size=(image.width, image.height))
        top = regions[0]

        out_rows.append(
            {
                "image_id": image_id,
                "assigned_rule_id": rule_id,
                "n_regions": len(regions),
                "top_region_source": top.source,
                "top_region_label": top.label,
                "top_region_box": json.dumps(top.box),
            }
        )

        # Before/after pairs make coordinate or masking errors visible before
        # those regions are used for causal perturbation on Day 5.
        if visualized < N_VISUALIZE:
            before = overlay_boxes(image, [top.box], labels=[f"top:{top.label}"], color="lime")
            after = mask_region(image, top.box, mode="black")
            save_figure(before, fig_dir / f"{image_id}_{rule_id}_before.png")
            save_figure(after, fig_dir / f"{image_id}_{rule_id}_after.png")
            visualized += 1

    out_csv = RESULTS_DIR / "region_extraction.csv"
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {out_csv}")
    print(f"Saved {visualized} before/after pairs to {fig_dir}")

    grid_fallback_rate = (out_df["top_region_source"] == "grid").mean()
    print(f"Grid-fallback rate (no model box at all): {grid_fallback_rate:.1%} ({(out_df['top_region_source']=='grid').sum()}/{len(out_df)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
