"""Dataset access and pilot-sample selection for ConstructionSite 10k."""

import random
from dataclasses import dataclass, field

from datasets import load_dataset

from xai_pilot.config import CLASS_PRIORITY, HF_DATASET_ID, RULE_TO_CLASS, SAMPLES_PER_CLASS, SEED

RULE_FIELDS = ["rule_1_violation", "rule_2_violation", "rule_3_violation", "rule_4_violation"]
RULE_SHORT_NAMES = {f: f.replace("_violation", "") for f in RULE_FIELDS}
COMPLIANT_ROUND_ROBIN = ["rule_1", "rule_2", "rule_3", "rule_4"]


@dataclass
class PilotSample:
    """Minimal metadata needed to reproduce one selected pilot observation.

    The image itself remains in the Hugging Face dataset. Storing only its
    identifier and experimental labels keeps the sample manifest small and
    allows every later script to retrieve the same image from the test split.
    """

    image_id: str
    primary_class: str
    violated_rule_ids: list[str] = field(default_factory=list)
    assigned_rule_id: str = ""


def classify_image(row: dict) -> tuple[str, list[str]]:
    """Map a dataset row's rule_1..4_violation fields to a pilot class.

    Returns (primary_class, violated_rule_ids). primary_class is the
    highest-priority violated class per CLASS_PRIORITY (ppe_violation >
    fall_hazard > struck_by_risk > compliant); violated_rule_ids lists every
    rule that was actually violated (a row can violate more than one).
    """
    violated_rule_ids = [
        field_name for field_name in RULE_FIELDS if row.get(field_name) is not None
    ]
    if not violated_rule_ids:
        return "compliant", []

    violated_classes = {RULE_TO_CLASS[r] for r in violated_rule_ids}
    for candidate_class in CLASS_PRIORITY:
        if candidate_class in violated_classes:
            return candidate_class, violated_rule_ids
    return "compliant", violated_rule_ids


def assign_rule_id(primary_class: str, violated_rule_ids: list[str], compliant_index: int = 0) -> str:
    """Pick the one rule a sample is tested against in Days 3-10.

    Violation classes have an unambiguous ground-truth rule (the first
    violated rule that maps to primary_class -- fall_hazard can come from
    rule_2 or rule_3, so the first one present wins). Compliant samples have
    no ground-truth rule, so they're round-robined evenly across all four
    rules to get balanced "model correctly says compliant" coverage instead
    of testing the same rule on every compliant image.
    """
    if primary_class == "compliant":
        return COMPLIANT_ROUND_ROBIN[compliant_index % len(COMPLIANT_ROUND_ROBIN)]
    for field_name in violated_rule_ids:
        if RULE_TO_CLASS[field_name] == primary_class:
            return RULE_SHORT_NAMES[field_name]
    raise ValueError(f"no violated rule matches primary_class={primary_class!r}")


def load_construction_site(split: str = "test", streaming: bool = True):
    """Load a ConstructionSite split using the repository's fixed dataset ID.

    Streaming is the default because the experiment usually needs only the
    selected image IDs and should not download the complete image corpus into
    memory before processing can begin.
    """
    return load_dataset(HF_DATASET_ID, split=split, streaming=streaming)


def select_balanced_sample(
    ds,
    n_per_class: int = SAMPLES_PER_CLASS,
    seed: int = SEED,
    classes: list[str] = CLASS_PRIORITY,
) -> list[PilotSample]:
    """Scan ds once, keep up to n_per_class rows per class, shuffle within class.

    ds is iterated once in streaming order; a reservoir of candidates per
    class is kept (capped at 10x n_per_class to bound memory) and the final
    selection is a seeded random sample from each class's reservoir.
    """
    rng = random.Random(seed)
    cap = n_per_class * 10
    reservoirs: dict[str, list[PilotSample]] = {c: [] for c in classes}

    # Build bounded candidate pools in one streaming pass. This is not a
    # textbook reservoir-sampling algorithm: once a class reaches `cap`, later
    # rows for that class are ignored. The seeded shuffle below therefore
    # randomizes a bounded prefix of each class while keeping memory usage
    # predictable for an image dataset.
    for row in ds:
        primary_class, violated_rule_ids = classify_image(row)
        if primary_class not in reservoirs:
            continue
        bucket = reservoirs[primary_class]
        if len(bucket) < cap:
            bucket.append(PilotSample(row["image_id"], primary_class, violated_rule_ids))
        if all(len(reservoirs[c]) >= cap for c in classes):
            break

    # Shuffle each class independently so class prevalence in the source
    # dataset cannot dominate the final balanced pilot subset.
    selected: list[PilotSample] = []
    for c in classes:
        bucket = reservoirs[c]
        rng.shuffle(bucket)
        selected.extend(bucket[:n_per_class])

    # Violation samples inherit a relevant violated rule. Compliant samples
    # have no violated rule, so assign them round-robin across all four rules
    # to avoid evaluating only one safety question on the compliant class.
    compliant_index = 0
    for sample in selected:
        sample.assigned_rule_id = assign_rule_id(
            sample.primary_class, sample.violated_rule_ids, compliant_index
        )
        if sample.primary_class == "compliant":
            compliant_index += 1
    return selected
