"""Day 6 deliverable: visual sparsity of the cross-attention attribution heatmap.

For each sample, computes a 24x24 cross-attention heatmap (attribution.py)
for the same phrase that produced Day 4's top-ranked region (the worker
phrase, or the rule's object phrase), then scores how concentrated that
heatmap is via metrics/sparsity.py. The 4 samples with no model box at all
(Day 4's grid-fallback cases) have no grounding-based phrase to attribute
to and are excluded from the aggregate, same as Day 4 flagged them.
"""

import sys

import pandas as pd

from xai_pilot.attribution import cross_attention_heatmap
from xai_pilot.config import FIGURES_DIR, RESULTS_DIR
from xai_pilot.data import load_construction_site
from xai_pilot.metrics.sparsity import regions_above_threshold, topk_mass_ratio
from xai_pilot.model import load_florence2
from xai_pilot.prompts import PERSON_PHRASE, RULE_4_PROXIMITY_PAIR, RULE_QUERIES
from xai_pilot.viz import overlay_boxes, overlay_heatmap, save_figure

N_VISUALIZE = 10
DETECTION_TASK = "<OPEN_VOCABULARY_DETECTION>"
THRESH = 0.5
TOPK_SMALL, TOPK_LARGE = 5, 20


def _phrase_for_top_region(rule_id: str, top_label: str) -> str:
    if top_label == "worker":
        return PERSON_PHRASE
    if rule_id == "rule_4":
        return RULE_4_PROXIMITY_PAIR[1]
    return RULE_QUERIES[rule_id][0]


def main() -> int:
    preds_df = pd.read_csv(RESULTS_DIR / "baseline_predictions.csv", dtype=str)
    regions_df = pd.read_csv(RESULTS_DIR / "region_extraction.csv", dtype=str)
    merged = preds_df.merge(regions_df, on=["image_id", "assigned_rule_id"])

    target_ids = set(merged["image_id"])
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

    fig_dir = FIGURES_DIR / "sparsity"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    visualized = 0
    for i, row in merged.iterrows():
        image_id = row["image_id"]
        rule_id = row["assigned_rule_id"]
        image = images_by_id[image_id]

        if row["top_region_source"] == "grid":
            out_rows.append(
                {
                    "image_id": image_id,
                    "assigned_rule_id": rule_id,
                    "primary_class": row["primary_class"],
                    "attributed_phrase": None,
                    "n_generated_tokens": 0,
                    "topk5_mass_ratio": float("nan"),
                    "topk20_mass_ratio": float("nan"),
                    "regions_above_0.5": float("nan"),
                    "excluded_reason": "grid_fallback_no_box",
                }
            )
            continue

        phrase = _phrase_for_top_region(rule_id, row["top_region_label"])
        result = cross_attention_heatmap(model, processor, image, DETECTION_TASK, text_input=phrase)

        out_rows.append(
            {
                "image_id": image_id,
                "assigned_rule_id": rule_id,
                "primary_class": row["primary_class"],
                "attributed_phrase": phrase,
                "n_generated_tokens": result.n_generated_tokens,
                "topk5_mass_ratio": topk_mass_ratio(result.heatmap, TOPK_SMALL),
                "topk20_mass_ratio": topk_mass_ratio(result.heatmap, TOPK_LARGE),
                "regions_above_0.5": regions_above_threshold(result.heatmap, THRESH),
                "excluded_reason": None,
            }
        )

        if visualized < N_VISUALIZE:
            heat_img = overlay_heatmap(image, result.heatmap, alpha=0.6, gamma=0.4)
            heat_img = overlay_boxes(heat_img, result.greedy_boxes, labels=[phrase], color="lime")
            save_figure(heat_img, fig_dir / f"{image_id}_{rule_id}_{phrase.replace(' ', '_')}.png")
            visualized += 1

        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(merged)} done...")

    out_csv = RESULTS_DIR / "visual_sparsity.csv"
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {out_csv}")
    print(f"Saved {visualized} heatmap overlays to {fig_dir}")

    scored = out_df[out_df["excluded_reason"].isna()]
    excluded = len(out_df) - len(scored)
    print()
    print(f"Excluded (grid fallback, no grounded phrase): {excluded}/{len(out_df)}")
    print(f"Mean topk5_mass_ratio: {scored['topk5_mass_ratio'].mean():.3f}")
    print(f"Mean topk20_mass_ratio: {scored['topk20_mass_ratio'].mean():.3f}")
    print(f"Mean regions_above_0.5: {scored['regions_above_0.5'].mean():.1f} / 576 cells")
    print()
    print("By assigned_rule_id (topk5_mass_ratio):")
    print(scored.groupby("assigned_rule_id")["topk5_mass_ratio"].mean())
    print()
    print("By attributed_phrase (topk5_mass_ratio):")
    print(scored.groupby("attributed_phrase")["topk5_mass_ratio"].mean())
    return 0


if __name__ == "__main__":
    sys.exit(main())
