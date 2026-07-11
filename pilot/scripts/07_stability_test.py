"""Day 7 deliverable: stability of Florence-2's grounding-proxy answers under resampling.

Florence-2's default decoding (beam search, no sampling) is deterministic,
so three literal reruns would be 100% stable by construction -- this reruns
each sample 3x with do_sample=True, num_beams=1, temperature=0.7 instead
(see metrics/stability.py), the actual stability test. Day 3's already-
logged beam-search answer is reported alongside as a separate "decoding
ceiling" reference, not mixed into the stability score.
"""

import argparse
import itertools
import json
import sys

import pandas as pd

from xai_pilot.config import FIGURES_DIR, REGION_RANKING, RESULTS_DIR
from xai_pilot.data import load_construction_site
from xai_pilot.metrics.stability import (
    answer_agreement_rate,
    mean_pairwise_centroid_distance,
    object_presence_rate,
    region_overlap_score,
    run_n_times,
)
from xai_pilot.model import load_florence2
from xai_pilot.prompts import RULE_OBJECT_LABEL, rule_object_labels
from xai_pilot.regions import REGION_RANKING_MODES, iou, standardize_regions
from xai_pilot.viz import overlay_boxes, save_figure

N_RERUNS = 3
N_VISUALIZE = 10
TEMPERATURE = 0.7
OBJECT_LABEL = RULE_OBJECT_LABEL


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


def main(
    region_ranking: str = REGION_RANKING,
    temperatures: list[float] | None = None,
    n_reruns: int | None = None,
    limit: int | None = None,
) -> int:
    """Quantify answer and region reproducibility across sampled reruns.

    `region_ranking` defaults to config.REGION_RANKING ("area", frozen pilot).
    With neither `temperatures` nor `n_reruns` given, the frozen single-point
    run (temperature 0.7, 3 reruns) reproduces byte-for-byte. Passing either
    enables SWEEP mode (Phase 2.3): each sample is rerun at every temperature,
    with per-temperature rows and the added object-presence and size-invariant
    centroid-distance stability columns, written to a suffixed CSV. `limit`
    subsets the samples to keep a dense sweep affordable.
    """
    sweep_mode = temperatures is not None or n_reruns is not None
    temps = temperatures if temperatures is not None else [TEMPERATURE]
    n = n_reruns if n_reruns is not None else N_RERUNS

    preds_df = pd.read_csv(RESULTS_DIR / "baseline_predictions.csv", dtype=str)
    if limit is not None:
        preds_df = preds_df.head(limit)
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

        for temperature in temps:
            # Sampling introduces controlled decoding variability. The saved
            # deterministic answer is retained only as an external reference.
            results = run_n_times(model, processor, image, rule_id, n=n, temperature=temperature)

            # Standardize every rerun independently because both the number and
            # geometry of returned grounding boxes may change across generations.
            top_regions = []
            for r in results:
                boxes = r.worker_boxes + r.object_boxes
                labels = ["worker"] * len(r.worker_boxes) + [OBJECT_LABEL[rule_id]] * len(r.object_boxes)
                regions = standardize_regions(
                    boxes,
                    labels,
                    image_size=(image.width, image.height),
                    region_ranking=region_ranking,
                    object_labels=rule_object_labels(rule_id),
                )
                top_regions.append(regions[0])

            sampled_answers = [r.answer for r in results]
            majority_answer = max(set(sampled_answers), key=sampled_answers.count)

            record = {
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
            if sweep_mode:
                # Object boxes present per rerun (size-invariant drift + presence).
                object_firsts = [r.object_boxes[0] for r in results if r.object_boxes]
                record["temperature"] = temperature
                record["n_reruns"] = n
                record["object_presence_rate"] = object_presence_rate(results)
                record["object_centroid_distance"] = mean_pairwise_centroid_distance(
                    object_firsts, image_size=(image.width, image.height)
                )
                # IoU here silently dropped appear/disappear reruns to NaN; the
                # presence rate above is what recovers that instability.
            out_rows.append(record)

            if visualized < N_VISUALIZE and temperature == temps[0]:
                boxes = [r.box for r in top_regions if r.source == "model"]
                labels = [f"run{j}" for j in range(len(boxes))]
                overlay = overlay_boxes(image, boxes, labels=labels, color="lime")
                save_figure(overlay, fig_dir / f"{image_id}_{rule_id}_reruns.png")
                visualized += 1

        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(preds_df)} done...")

    parts = [p for p in (
        region_ranking if region_ranking != "area" else "",
        "sweep" if sweep_mode else "",
    ) if p]
    suffix = ("_" + "_".join(parts)) if parts else ""
    out_csv = RESULTS_DIR / f"stability{suffix}.csv"
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Ranking policy: {region_ranking} | reruns: {n} | temps: {temps}")
    print(f"Wrote {len(out_df)} rows to {out_csv}")
    print(f"Saved {visualized} rerun-overlay images to {fig_dir}")

    print()
    if sweep_mode:
        print("Stability vs. temperature (mean over samples):")
        curve = out_df.groupby("temperature").agg(
            answer_agreement=("answer_agreement_rate", "mean"),
            object_overlap=("object_region_overlap_score", "mean"),
            object_centroid_dist=("object_centroid_distance", "mean"),
            object_presence=("object_presence_rate", "mean"),
        )
        print(curve.to_string(float_format=lambda x: f"{x:.3f}"))
        print()
        print(f"Mean object_presence_rate (1.0 = object detected every rerun): "
              f"{out_df['object_presence_rate'].mean():.3f}")
        print(f"Samples with a flickering object (presence strictly between 0 and 1): "
              f"{int(((out_df['object_presence_rate'] > 0) & (out_df['object_presence_rate'] < 1)).sum())}")
    else:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region-ranking",
        choices=REGION_RANKING_MODES,
        default=REGION_RANKING,
        help="Candidate-region ranking policy (default: config.REGION_RANKING).",
    )
    parser.add_argument(
        "--temperatures",
        type=lambda s: [float(x) for x in s.split(",")],
        default=None,
        help="Comma-separated sampling temperatures to sweep, e.g. 0.3,0.5,0.7,1.0 (enables sweep mode).",
    )
    parser.add_argument(
        "--n-reruns",
        type=int,
        default=None,
        help="Reruns per sample per temperature (enables sweep mode; scale run uses >=10).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples (keeps a dense sweep affordable).",
    )
    args = parser.parse_args()
    sys.exit(main(
        region_ranking=args.region_ranking,
        temperatures=args.temperatures,
        n_reruns=args.n_reruns,
        limit=args.limit,
    ))
