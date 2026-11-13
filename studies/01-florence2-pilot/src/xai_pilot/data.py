"""Dataset access and pilot-sample selection for ConstructionSite 10k."""

import random
from dataclasses import dataclass, field

from datasets import Image as HFImage, load_dataset

from xai_pilot.config import (
    CLASS_PRIORITY,
    HF_DATASET_ID,
    LOCAL_DATASET_DIR,
    RULE_TO_CLASS,
    SAMPLES_PER_CLASS,
    SEED,
)

RULE_FIELDS = ["rule_1_violation", "rule_2_violation", "rule_3_violation", "rule_4_violation"]
RULE_SHORT_NAMES = {f: f.replace("_violation", "") for f in RULE_FIELDS}
COMPLIANT_ROUND_ROBIN = ["rule_1", "rule_2", "rule_3", "rule_4"]
# Rules used for compliant images with no confirmed excavator context: rule_4
# (struck-by) is excluded because it would be a vacuous test without an
# excavator in frame. See assign_compliant_rule.
COMPLIANT_NON_EXCAVATOR = ["rule_1", "rule_2", "rule_3"]


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
    # Phase 1.2: every rule this image is tested against. Under priority
    # labeling this is always [assigned_rule_id] (one rule); under multi-label
    # labeling a multi-hazard image carries every rule it violates.
    assigned_rule_ids: list[str] = field(default_factory=list)
    # Phase 1.3: True when a compliant image was assigned a rule whose
    # triggering context could not be confirmed present in the scene (so the
    # test may be vacuous). Only meaningful under context_matched assignment.
    context_absent: bool = False


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


def image_classes(row: dict) -> list[str]:
    """Return every violated class for a row, highest-priority first.

    Where classify_image collapses a multi-violation row to a single
    primary_class (discarding the rest), this returns the *full* set of
    violated classes -- exactly the information priority-collapse throws away.
    A row with a rule_1 (PPE) and a rule_4 (struck-by) violation returns
    ["ppe_violation", "struck_by_risk"]; a clean row returns ["compliant"].
    Duplicate classes (rule_2 and rule_3 both map to fall_hazard) appear once.
    """
    violated_rule_ids = [f for f in RULE_FIELDS if row.get(f) is not None]
    if not violated_rule_ids:
        return ["compliant"]
    violated_classes = {RULE_TO_CLASS[r] for r in violated_rule_ids}
    return [c for c in CLASS_PRIORITY if c in violated_classes]


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


def scene_has_excavator(row: dict) -> bool:
    """True when the dataset's excavator metadata lists at least one box.

    The ConstructionSite parquet carries an ``excavator`` column of bounding
    boxes (empty list when absent). This is the one rule-context signal
    reliably derivable from metadata alone, and it gates rule_4 (struck-by).
    """
    return bool(row.get("excavator"))


def assign_compliant_rule(
    has_excavator: bool, compliant_index: int, mode: str = "round_robin"
) -> tuple[str, bool]:
    """Choose which rule a compliant image is tested against.

    Returns ``(rule_id, context_absent)``.

    - ``"round_robin"`` (frozen pilot): cycle all four rules evenly, without
      inspecting the scene. ``context_absent`` is always False because context
      is not assessed.
    - ``"context_matched"`` (Phase 1.3): route a scene with an excavator to
      rule_4 (its struck-by context is confirmed present); never assign the
      vacuous rule_4 to a scene without an excavator. Scenes without an
      excavator round-robin over rules 1--3 and are flagged
      ``context_absent=True``, because metadata cannot confirm PPE/height/edge
      context -- that requires the grounding pre-check deferred to a later
      phase. The flag lets those assignments be reported separately rather than
      silently inflating baseline compliant accuracy.
    """
    if mode == "round_robin":
        return COMPLIANT_ROUND_ROBIN[compliant_index % len(COMPLIANT_ROUND_ROBIN)], False
    if mode == "context_matched":
        if has_excavator:
            return "rule_4", False
        rule = COMPLIANT_NON_EXCAVATOR[compliant_index % len(COMPLIANT_NON_EXCAVATOR)]
        return rule, True
    raise ValueError(f"unknown compliant_assignment mode={mode!r}")


def _local_parquet_files() -> dict[str, list[str]]:
    """Map split name -> local parquet shard paths, if the download exists.

    Returns an empty mapping when no local snapshot is present, so callers
    transparently fall back to Hub streaming.
    """
    if not LOCAL_DATASET_DIR.is_dir():
        return {}
    mapping: dict[str, list[str]] = {}
    for split, pattern in (("test", "test*.parquet"), ("train", "train*.parquet")):
        shards = sorted(LOCAL_DATASET_DIR.glob(pattern))
        if shards:
            mapping[split] = [str(p) for p in shards]
    return mapping


def load_construction_site(split: str = "test", streaming: bool = True):
    """Load a ConstructionSite split, preferring a local copy over the Hub.

    If the dataset has been downloaded to LOCAL_DATASET_DIR (parquet shards),
    it is read from local disk so GPU/cluster runs are bound by compute rather
    than by streaming I/O from the Hub. When no local snapshot exists, it falls
    back to streaming the dataset from the Hub by its fixed ID.

    Streaming stays the default access mode either way: the pilot usually needs
    only selected image IDs, not the whole corpus resident in memory.
    """
    local_files = _local_parquet_files()
    if split in local_files:
        ds = load_dataset(
            "parquet", data_files={split: local_files[split]}, split=split, streaming=streaming
        )
        # The raw parquet does not restore the Hub's Image feature, so the
        # image column decodes to a {bytes, path} dict; cast it back to PIL.
        return ds.cast_column("image", HFImage())
    return load_dataset(HF_DATASET_ID, split=split, streaming=streaming)


def select_balanced_sample(
    ds,
    n_per_class: int = SAMPLES_PER_CLASS,
    seed: int = SEED,
    classes: list[str] = CLASS_PRIORITY,
    labeling: str = "priority",
    compliant_assignment: str = "round_robin",
) -> list[PilotSample]:
    """Scan ds once and pick a seeded, class-balanced pilot subset.

    `labeling` selects the class-assignment policy:

    - ``"priority"`` (default, frozen pilot): each image belongs to exactly one
      class -- its highest-priority violation -- and is tested against exactly
      one rule. Reproduces the recorded pilot manifest byte-for-byte.
    - ``"multilabel"`` (Phase 1.2): an image belongs to *every* class it
      violates and is tested against *every* rule it violates. This recovers
      scarce classes (notably struck_by_risk) that priority-collapse hides
      behind a higher-priority violation on the same image.

    `compliant_assignment` selects how compliant images pick a rule to be tested
    against: ``"round_robin"`` (default, frozen) cycles all four rules evenly;
    ``"context_matched"`` (Phase 1.3) routes excavator scenes to rule_4 and
    avoids assigning the vacuous rule_4 to excavator-free scenes. See
    assign_compliant_rule.
    """
    if labeling == "priority":
        return _select_priority(ds, n_per_class, seed, classes, compliant_assignment)
    if labeling == "multilabel":
        return _select_multilabel(ds, n_per_class, seed, classes, compliant_assignment)
    raise ValueError(f"unknown labeling={labeling!r}; expected 'priority' or 'multilabel'")


def _select_priority(ds, n_per_class, seed, classes, compliant_assignment="round_robin") -> list[PilotSample]:
    """Frozen pilot sampler: one class and one rule per image."""
    rng = random.Random(seed)
    cap = n_per_class * 10
    reservoirs: dict[str, list[PilotSample]] = {c: [] for c in classes}
    # Excavator presence is captured only when context matching needs it, so
    # the frozen round-robin path does no extra per-row work.
    excavator_by_id: dict[str, bool] = {}

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
            if compliant_assignment == "context_matched" and primary_class == "compliant":
                excavator_by_id[row["image_id"]] = scene_has_excavator(row)
        if all(len(reservoirs[c]) >= cap for c in classes):
            break

    # Shuffle each class independently so class prevalence in the source
    # dataset cannot dominate the final balanced pilot subset.
    selected: list[PilotSample] = []
    for c in classes:
        bucket = reservoirs[c]
        rng.shuffle(bucket)
        selected.extend(bucket[:n_per_class])

    _assign_rules(selected, compliant_assignment, excavator_by_id)
    return selected


def _assign_rules(samples, compliant_assignment, excavator_by_id) -> None:
    """Assign each sample's rule(s) in place.

    Violation samples inherit their ground-truth rule. Compliant samples have
    no violated rule, so they are assigned per `compliant_assignment` (frozen
    round-robin, or Phase 1.3 context matching).
    """
    compliant_index = 0
    for sample in samples:
        if sample.primary_class == "compliant":
            has_excavator = excavator_by_id.get(sample.image_id, False)
            rule, context_absent = assign_compliant_rule(
                has_excavator, compliant_index, mode=compliant_assignment
            )
            sample.assigned_rule_id = rule
            sample.context_absent = context_absent
            compliant_index += 1
        else:
            sample.assigned_rule_id = assign_rule_id(
                sample.primary_class, sample.violated_rule_ids
            )
        sample.assigned_rule_ids = [sample.assigned_rule_id]


def _select_multilabel(ds, n_per_class, seed, classes, compliant_assignment="round_robin") -> list[PilotSample]:
    """Multi-label sampler: an image counts toward -- and is tested against --
    every class/rule it violates.

    An image that violates two rules is drawn once (deduplicated by image_id)
    and carries every violated rule in ``assigned_rule_ids``; compliant images
    are assigned a single rule per `compliant_assignment` (round-robin, or
    Phase 1.3 context matching). Each class reservoir is filled independently,
    so a class's coverage floor is guaranteed even when its images are shared
    with a higher-priority class.
    """
    rng = random.Random(seed)
    cap = n_per_class * 10
    # Reservoir stores (image_id, violated_rule_ids_tuple); an image with two
    # violated classes lands in two reservoirs but keeps identical rule data.
    reservoirs: dict[str, list[tuple[str, tuple[str, ...]]]] = {c: [] for c in classes}
    excavator_by_id: dict[str, bool] = {}

    for row in ds:
        violated = tuple(f for f in RULE_FIELDS if row.get(f) is not None)
        for c in image_classes(row):
            if c not in reservoirs:
                continue
            bucket = reservoirs[c]
            if len(bucket) < cap:
                bucket.append((row["image_id"], violated))
                if compliant_assignment == "context_matched" and not violated:
                    excavator_by_id[row["image_id"]] = scene_has_excavator(row)
        if all(len(reservoirs[c]) >= cap for c in classes):
            break

    # Select per class (seeded shuffle in the same class order as priority),
    # deduplicating images across classes while unioning their violated rules.
    chosen: dict[str, set[str]] = {}
    order: list[str] = []
    for c in classes:
        bucket = reservoirs[c]
        rng.shuffle(bucket)
        for image_id, violated in bucket[:n_per_class]:
            if image_id not in chosen:
                chosen[image_id] = set(violated)
                order.append(image_id)
            else:
                chosen[image_id].update(violated)

    samples: list[PilotSample] = []
    compliant_index = 0
    for image_id in order:
        violated = [f for f in RULE_FIELDS if f in chosen[image_id]]
        context_absent = False
        if not violated:
            primary_class = "compliant"
            assigned_rule_id, context_absent = assign_compliant_rule(
                excavator_by_id.get(image_id, False), compliant_index, mode=compliant_assignment
            )
            assigned_rule_ids = [assigned_rule_id]
            compliant_index += 1
        else:
            violated_classes = {RULE_TO_CLASS[r] for r in violated}
            primary_class = next(c for c in CLASS_PRIORITY if c in violated_classes)
            # Tested against every rule it violates, in canonical rule order.
            assigned_rule_ids = [RULE_SHORT_NAMES[r] for r in violated]
            assigned_rule_id = assigned_rule_ids[0]
        samples.append(
            PilotSample(
                image_id=image_id,
                primary_class=primary_class,
                violated_rule_ids=violated,
                assigned_rule_id=assigned_rule_id,
                assigned_rule_ids=assigned_rule_ids,
                context_absent=context_absent,
            )
        )
    return samples
