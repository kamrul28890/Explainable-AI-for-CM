"""Scale-up Phase 2.5: multi-hazard bounded completeness.

The proposal's core promise was that an explanation should capture *every*
hazard an image exhibits, not just one. That is only testable with the
multi-label data (Phase 1.2). This script finds test-split images that violate
two or more rules and, for each violated rule independently, masks that rule's
own explanation region (rule-aware ranking, Phase 1.1) and checks whether the
rule's answer flips -- worker-loss-corrected (Phase 1.4). An image is
multi-hazard "complete" only if masking each hazard's region flips that
hazard's answer.

This also exercises the (image, rule) multi-testing that the full Phase 5 scale
run generalizes; here it is scoped to the multi-hazard subset only.
"""

import sys

import pandas as pd

from xai_pilot.data import RULE_FIELDS, load_construction_site
from xai_pilot.inference import answer_rule
from xai_pilot.metrics.completeness import classify_sample_worker_loss_corrected, multi_hazard_verdict
from xai_pilot.metrics.descriptive_accuracy import evaluate
from xai_pilot.model import load_florence2
from xai_pilot.prompts import RULE_OBJECT_LABEL, rule_object_labels
from xai_pilot.regions import standardize_regions
from xai_pilot.config import RESULTS_DIR


def _violated_rules(row) -> list[str]:
    """Rule short-names (rule_1..rule_4) an image violates."""
    return [f.replace("_violation", "") for f in RULE_FIELDS if row.get(f) is not None]


def _rule_supported(model, processor, image, rule_id) -> tuple[str, bool]:
    """Return (top_region_source, worker-loss-corrected supported) for one rule."""
    baseline = answer_rule(model, processor, image, rule_id)
    boxes = baseline.worker_boxes + baseline.object_boxes
    labels = ["worker"] * len(baseline.worker_boxes) + [RULE_OBJECT_LABEL[rule_id]] * len(
        baseline.object_boxes
    )
    regions = standardize_regions(
        boxes, labels, image_size=(image.width, image.height),
        region_ranking="rule_aware", object_labels=rule_object_labels(rule_id),
    )
    result = evaluate(model, processor, image, rule_id, baseline, regions)
    source = regions[0].source
    verdict = classify_sample_worker_loss_corrected(
        source,
        result.answer_changed_top1,
        result.answer_changed_top2,
        result.flip_due_to_worker_loss_top1,
        result.flip_due_to_worker_loss_top2,
    )
    return source, verdict == "explanation_supported"


def main() -> int:
    print("Scanning test split for multi-hazard (2+ rule) images...")
    ds = load_construction_site(split="test", streaming=True)
    multi = []  # (image, [rules])
    for row in ds:
        rules = _violated_rules(row)
        if len(rules) >= 2:
            multi.append((row["image"], row["image_id"], rules))
    print(f"Found {len(multi)} multi-hazard images.")

    print("Loading Florence-2-base-ft...")
    model, processor = load_florence2()

    rows = []
    for k, (image, image_id, rules) in enumerate(multi):
        supported = {}
        for rule_id in rules:
            _, ok = _rule_supported(model, processor, image, rule_id)
            supported[rule_id] = ok
        verdict = multi_hazard_verdict(supported)
        rows.append(
            {
                "image_id": image_id,
                "rules": "|".join(rules),
                "n_hazards": len(rules),
                "n_supported": sum(supported.values()),
                "verdict": verdict,
            }
        )
        if (k + 1) % 5 == 0:
            print(f"{k + 1}/{len(multi)} done...")

    out_df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "multihazard_completeness.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_csv}")
    print("\nMulti-hazard completeness verdict distribution:")
    print(out_df["verdict"].value_counts().to_string())
    if len(out_df):
        complete = (out_df["verdict"] == "complete").mean()
        print(f"\nFully-complete (every hazard's region load-bearing): {complete:.1%} "
              f"({int((out_df['verdict']=='complete').sum())}/{len(out_df)})")
        print(f"Mean hazards supported per image: {out_df['n_supported'].mean():.2f} "
              f"of {out_df['n_hazards'].mean():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
