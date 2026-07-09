"""Day 2 deliverable: a fixed, balanced, reproducible pilot subset.

Streams the ConstructionSite test split once, buckets rows by
classify_image's primary_class, and writes a seeded random sample of
SAMPLES_PER_CLASS per class to data/pilot_samples.csv, plus the rule
query phrasings to data/safety_prompts.json.
"""

import argparse
import csv
import json
import sys

from xai_pilot.config import (
    CLASS_PRIORITY,
    COMPLIANT_ASSIGNMENT,
    DATA_DIR,
    LABELING,
    RULE_TO_CLASS,
    SAMPLES_PER_CLASS,
    SEED,
)
from xai_pilot.data import load_construction_site, select_balanced_sample
from xai_pilot.prompts import RULE_4_PROXIMITY_PAIR, RULE_QUERIES


def _sample_class_set(sample) -> set[str]:
    """Every class an image is labeled for (from its violated rules)."""
    if not sample.violated_rule_ids:
        return {"compliant"}
    return {RULE_TO_CLASS[r] for r in sample.violated_rule_ids}


def main(labeling: str = LABELING, compliant_assignment: str = COMPLIANT_ASSIGNMENT) -> int:
    """Create the reproducible sample manifest and prompt configuration.

    Defaults reproduce the frozen pilot: labeling="priority" and
    compliant_assignment="round_robin" write the original 5-column
    pilot_samples.csv byte-for-byte. Any non-default policy writes an expanded
    (assigned_rule_ids + context_absent), mode-suffixed manifest so the frozen
    file is never overwritten.
    """
    frozen = labeling == "priority" and compliant_assignment == "round_robin"
    print(
        f"Streaming ConstructionSite test split, target {SAMPLES_PER_CLASS}/class "
        f"(labeling={labeling}, compliant={compliant_assignment})..."
    )
    ds = load_construction_site(split="test", streaming=True)
    samples = select_balanced_sample(
        ds, n_per_class=SAMPLES_PER_CLASS, seed=SEED,
        labeling=labeling, compliant_assignment=compliant_assignment,
    )

    if labeling == "multilabel":
        # An image counts toward every class it violates, so per-class coverage
        # is the meaningful tally (it can exceed len(samples) via shared images).
        coverage = {c: 0 for c in CLASS_PRIORITY}
        for s in samples:
            for c in _sample_class_set(s):
                coverage[c] += 1
        print(f"Selected {len(samples)} unique images.")
        print("Per-class coverage (images labeled for each class):", coverage)
    else:
        counts = {c: 0 for c in CLASS_PRIORITY}
        for s in samples:
            counts[s.primary_class] += 1
        print("Class counts:", counts)
    if compliant_assignment == "context_matched":
        n_absent = sum(1 for s in samples if s.context_absent)
        print(f"Compliant assignments flagged context_absent: {n_absent}")

    # Store lists as pipe-delimited values because a sample can violate more
    # than one rule while CSV still requires one scalar value per cell.
    if frozen:
        out_csv = DATA_DIR / "pilot_samples.csv"
        header = ["image_id", "split", "primary_class", "violated_rule_ids", "assigned_rule_id"]
        rows = (
            [s.image_id, "test", s.primary_class, "|".join(s.violated_rule_ids), s.assigned_rule_id]
            for s in samples
        )
    else:
        parts = [p for p in (
            labeling if labeling != "priority" else "",
            compliant_assignment if compliant_assignment != "round_robin" else "",
        ) if p]
        out_csv = DATA_DIR / f"pilot_samples_{'_'.join(parts)}.csv"
        header = [
            "image_id", "split", "primary_class", "violated_rule_ids",
            "assigned_rule_id", "assigned_rule_ids", "context_absent",
        ]
        rows = (
            [
                s.image_id, "test", s.primary_class, "|".join(s.violated_rule_ids),
                s.assigned_rule_id, "|".join(s.assigned_rule_ids), int(s.context_absent),
            ]
            for s in samples
        )
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote {len(samples)} rows to {out_csv}")

    out_prompts = DATA_DIR / "safety_prompts.json"
    # Snapshot the exact prompt phrases beside the sample manifest so later
    # results remain interpretable if source constants are changed.
    prompts_payload = {
        "rule_queries": RULE_QUERIES,
        "rule_4_proximity_pair": list(RULE_4_PROXIMITY_PAIR),
    }
    with open(out_prompts, "w", encoding="utf-8") as f:
        json.dump(prompts_payload, f, indent=2)
    print(f"Wrote prompts to {out_prompts}")

    underfilled = [c for c, n in counts.items() if n < SAMPLES_PER_CLASS]
    if underfilled:
        # struck_by_risk (rule_4 / excavator proximity) is a genuinely rare
        # label in ConstructionSite 10k: only 13 examples in the full 3,004
        # row test split (confirmed by a full-split scan), 34 in the 7,009
        # row train split. This is a dataset property, not a sampling bug --
        # documented as a pilot limitation rather than padded across splits.
        print(f"NOTE: classes below target count (dataset is genuinely scarce there): {underfilled}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labeling",
        choices=["priority", "multilabel"],
        default=LABELING,
        help="Class-labeling policy (default: config.LABELING).",
    )
    parser.add_argument(
        "--compliant-assignment",
        choices=["round_robin", "context_matched"],
        default=COMPLIANT_ASSIGNMENT,
        help="Compliant-image rule assignment policy (default: config.COMPLIANT_ASSIGNMENT).",
    )
    args = parser.parse_args()
    sys.exit(main(labeling=args.labeling, compliant_assignment=args.compliant_assignment))
