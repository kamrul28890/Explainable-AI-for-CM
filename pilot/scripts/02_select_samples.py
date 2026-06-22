"""Day 2 deliverable: a fixed, balanced, reproducible pilot subset.

Streams the ConstructionSite test split once, buckets rows by
classify_image's primary_class, and writes a seeded random sample of
SAMPLES_PER_CLASS per class to data/pilot_samples.csv, plus the rule
query phrasings to data/safety_prompts.json.
"""

import csv
import json
import sys

from xai_pilot.config import CLASS_PRIORITY, DATA_DIR, SAMPLES_PER_CLASS, SEED
from xai_pilot.data import load_construction_site, select_balanced_sample
from xai_pilot.prompts import RULE_4_PROXIMITY_PAIR, RULE_QUERIES


def main() -> int:
    print(f"Streaming ConstructionSite test split, target {SAMPLES_PER_CLASS}/class...")
    ds = load_construction_site(split="test", streaming=True)
    samples = select_balanced_sample(ds, n_per_class=SAMPLES_PER_CLASS, seed=SEED)

    counts = {c: 0 for c in CLASS_PRIORITY}
    for s in samples:
        counts[s.primary_class] += 1
    print("Class counts:", counts)

    out_csv = DATA_DIR / "pilot_samples.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_id", "split", "primary_class", "violated_rule_ids", "assigned_rule_id"]
        )
        for s in samples:
            writer.writerow(
                [s.image_id, "test", s.primary_class, "|".join(s.violated_rule_ids), s.assigned_rule_id]
            )
    print(f"Wrote {len(samples)} rows to {out_csv}")

    out_prompts = DATA_DIR / "safety_prompts.json"
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
    sys.exit(main())
