"""Day 7 deliverable: stability of Florence-2's grounding-proxy answers under resampling.

Florence-2's default decoding (beam search, no sampling) is deterministic,
so three literal reruns would be 100% stable by construction -- this reruns
each sample 3x with do_sample=True, num_beams=1, temperature=0.7 instead
(see metrics/stability.py), the actual stability test. Day 3's already-
logged beam-search answer is reported alongside as a separate "decoding
ceiling" reference, not mixed into the stability score.
"""

import itertools
import json
import sys

import pandas as pd

from xai_pilot.config import FIGURES_DIR, RESULTS_DIR
from xai_pilot.data import load_construction_site
from xai_pilot.metrics.stability import answer_agreement_rate, region_overlap_score, run_n_times
from xai_pilot.model import load_florence2
from xai_pilot.regions import iou, standardize_regions
from xai_pilot.viz import overlay_boxes, save_figure

N_RERUNS = 3
N_VISUALIZE = 10
TEMPERATURE = 0.7
OBJECT_LABEL = {"rule_1": "hard hat", "rule_2": "harness", "rule_3": "guardrail", "rule_4": "excavator"}


def _object_region_overlap(results) -> float:
    """Pairwise IoU of the rule's safety-object box specifically.

    Distinct from region_overlap_score's top-1 region, which Day 4's
    area-based ranking usually fills with the (much larger) worker box --
    see day7_findings.md for why that makes top-1 overlap understate the
    instability of the actually safety-relevant detection.
    """
    object_firsts = [r.object_boxes[0] for r in results if r.object_boxes]
    if len(object_firsts) < len(results) or len(object_firsts) < 2:
        return float("nan")
    pairs = list(itertools.combinations(object_firsts, 2))
    return sum(iou(a, b) for a, b in pairs) / len(pairs)


def main() -> int:
    """Quantify answer and region reproducibility across sampled reruns."""
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

    fig_dir = FIGURES_DIR / "stability"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    visualized = 0
    for i, row in preds_df.iterrows():
        image_id = row["image_id"]
        rule_id = row["assigned_rule_id"]
        image = images_by_id[image_id]
        deterministic_answer = row["answer"]

        # Sampling introduces controlled decoding variability. The saved
        # deterministic answer is retained only as an external reference.
        results = run_n_times(model, processor, image, rule_id, n=N_RERUNS, temperature=TEMPERATURE)

        # Standardize every rerun independently because both the number and
        # geometry of returned grounding boxes may change across generations.
        top_regions = []
        for r in results:
            boxes = r.worker_boxes + r.object_boxes
            labels = ["worker"] * len(r.worker_boxes) + [OBJECT_LABEL[rule_id]] * len(r.object_boxes)
            regions = standardize_regions(boxes, labels, image_size=(image.width, image.height))
            top_regions.append(regions[0])

        sampled_answers = [r.answer for r in results]
        # With three reruns a majority always exists. `max` is sufficient here;
        # the odd rerun count avoids ambiguous two-way ties.
        majority_answer = max(set(sampled_answers), key=sampled_answers.count)

        out_rows.append(
            {
                "image_id": image_id,
                "assigned_rule_id": rule_id,
                "primary_class": row["primary_class"],
                "deterministic_answer": deterministic_answer,
                "sampled_answers": json.dumps(sampled_answers),
                "answer_agreement_rate": answer_agreement_rate(results),
                "majority_sampled_answer": majority_answer,
                "deterministic_matches_majority": deterministic_answer == majority_answer,
                "top_region_overlap_score": region_overlap_score(top_regions),
                "object_region_overlap_score": _object_region_overlap(results),
            }
        )

        if visualized < N_VISUALIZE:
            boxes = [r.box for r in top_regions if r.source == "model"]
            labels = [f"run{j}" for j in range(len(boxes))]
            overlay = overlay_boxes(image, boxes, labels=labels, color="lime")
            save_figure(overlay, fig_dir / f"{image_id}_{rule_id}_reruns.png")
            visualized += 1

        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(preds_df)} done...")

    out_csv = RESULTS_DIR / "stability.csv"
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {out_csv}")
    print(f"Saved {visualized} rerun-overlay images to {fig_dir}")

    print()
    print(f"Mean answer_agreement_rate: {out_df['answer_agreement_rate'].mean():.3f}")
    print(f"Deterministic-matches-sampled-majority rate: {out_df['deterministic_matches_majority'].mean():.1%}")
    print(f"Mean top_region_overlap_score (excl. nan): {out_df['top_region_overlap_score'].mean():.3f}")
    print(f"Mean object_region_overlap_score (excl. nan): {out_df['object_region_overlap_score'].mean():.3f}")
    print()
    print("By assigned_rule_id (answer_agreement_rate):")
    print(out_df.groupby("assigned_rule_id")["answer_agreement_rate"].mean())
    print()
    print("By assigned_rule_id (object_region_overlap_score):")
    print(out_df.groupby("assigned_rule_id")["object_region_overlap_score"].mean())
    return 0


if __name__ == "__main__":
    sys.exit(main())
