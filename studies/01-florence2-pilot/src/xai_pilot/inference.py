"""VQA-via-grounding: answer a safety rule for one image using Florence-2.

Florence-2-base-ft has no free-form VQA task token, so each rule is answered
by grounding a phrase instead of asking a yes/no question (see
pilot-plan.md, Design Decision #1). Rule 4 is a proximity check between two
independently grounded objects (worker, excavator).

Rules 1-3 are also person-relative proximity checks, not scene-level
presence checks: an earlier version asked "does <safety object> exist
anywhere in this image?" via a single <OPEN_VOCABULARY_DETECTION> call, but
that measures scene-level object existence, not whether a *specific* worker
lacks the object -- which is what the dataset's rule_X_violation labels
actually mean. Most multi-worker images have at least one compliant worker,
so the scene-level check almost always found a box and answered
"compliant" regardless of ground truth (documented in
results/day3_findings.md: 3% sensitivity to real violations). The fix:
detect every worker and every instance of the safety object independently,
then flag "violation" if any detected worker has no nearby/overlapping
safety-object box -- mirroring the dataset's own semantics that one
non-compliant worker is enough to mark the image violated.
"""

import time
from dataclasses import dataclass, field

from PIL import Image

from xai_pilot.model import run_task
from xai_pilot.prompts import PERSON_PHRASE, RULE_4_PROXIMITY_PAIR, RULE_QUERIES, RuleId
from xai_pilot.regions import all_boxes_covered, boxes_overlap_or_close, fraction_covered

Box = tuple[float, float, float, float]

DETECTION_TASK = "<OPEN_VOCABULARY_DETECTION>"
PROXIMITY_FRACTION = 0.05  # blind-spot distance threshold as a fraction of max(image dim)

# Worn PPE (rules 1-2) must be tight to the worker's body. Guardrails (rule 3)
# are structural and can protect a worker from further away (e.g. running
# along an excavation edge rather than touching them), so rule 3 gets a
# looser threshold. Both are fractions of max(image dimension).
PRESENCE_PROXIMITY_FRACTION: dict[str, float] = {
    "rule_1": 0.08,
    "rule_2": 0.08,
    "rule_3": 0.20,
}


@dataclass
class AnswerResult:
    """Normalized output shared by every metric in the pilot.

    `boxes` contains all regions used by downstream region ranking, while the
    worker/object lists remain separate so robustness and stability can detect
    semantically important failures such as losing the worker detection.

    `graded_score` (Phase 2.1) is a continuous [0, 1] compliance/safety signal
    (higher = safer): for presence rules it is the fraction of workers with a
    nearby safety object; for rule_4 it is the fraction of workers *not* near an
    excavator. It is NaN when there is no worker to score. Masking/perturbation
    metrics compare graded_score deltas to measure effect magnitude rather than
    only a boolean answer flip, recovering statistical power.
    """

    answer: str
    boxes: list[Box] = field(default_factory=list)
    confidence: float = float("nan")
    inference_ms: float = 0.0
    worker_boxes: list[Box] = field(default_factory=list)
    object_boxes: list[Box] = field(default_factory=list)
    graded_score: float = float("nan")


def _presence_verdict(
    worker_boxes: list[Box], object_boxes: list[Box], distance_threshold: float
) -> tuple[str, float]:
    """Pure verdict for a presence rule: (answer, graded_score).

    Compliant only when every detected worker has a nearby safety object (graded
    score 1.0). With no workers detected, falls back to the scene-level check
    (compliant iff the object exists anywhere) and returns a NaN graded score,
    since a per-worker fraction is undefined. Preserves the frozen boolean
    answer exactly (compliant iff the graded fraction is 1.0).
    """
    if not worker_boxes:
        return ("compliant" if object_boxes else "violation"), float("nan")
    frac = fraction_covered(worker_boxes, object_boxes, distance_threshold=distance_threshold)
    return ("compliant" if frac >= 1.0 else "violation"), frac


def _rule4_verdict(
    worker_boxes: list[Box], excavator_boxes: list[Box], distance_threshold: float
) -> tuple[str, float]:
    """Pure verdict for rule_4 proximity: (answer, graded_score).

    Hazard when any worker is near any excavator. graded_score is the *safe*
    fraction (1 - fraction of workers near an excavator), so higher stays safer
    across all rules; NaN when there is no worker to score. Preserves the frozen
    boolean answer exactly (hazard iff any worker is near an excavator).
    """
    near_frac = fraction_covered(worker_boxes, excavator_boxes, distance_threshold=distance_threshold)
    hazard = any(
        boxes_overlap_or_close(w, e, distance_threshold=distance_threshold)
        for w in worker_boxes
        for e in excavator_boxes
    )
    graded = float("nan") if near_frac != near_frac else 1.0 - near_frac
    return ("hazard" if hazard else "safe"), graded


def _detect(model, processor, image: Image.Image, phrase: str, **run_kwargs):
    """Ground one noun phrase and normalize Florence-2 boxes to tuples."""
    _, parsed, confidence = run_task(
        model, processor, image, DETECTION_TASK, text_input=phrase, **run_kwargs
    )
    det = parsed[DETECTION_TASK]
    boxes = [tuple(b) for b in det["bboxes"]]
    return boxes, confidence


def graded_score_for(
    rule_id: RuleId,
    worker_boxes: list[Box],
    object_boxes: list[Box],
    image_size: tuple[int, int],
) -> float:
    """Recompute the continuous graded score from already-detected boxes.

    Lets a metric that reconstructs a baseline from saved worker/object boxes
    (rather than re-running the model) obtain the same graded score answer_rule
    would have produced, using the rule's own resolution-scaled threshold.
    """
    width, height = image_size
    if rule_id == "rule_4":
        threshold = PROXIMITY_FRACTION * max(width, height)
        return _rule4_verdict(worker_boxes, object_boxes, distance_threshold=threshold)[1]
    threshold = PRESENCE_PROXIMITY_FRACTION[rule_id] * max(width, height)
    return _presence_verdict(worker_boxes, object_boxes, distance_threshold=threshold)[1]


def answer_rule_text_ablated(
    model,
    processor,
    image: Image.Image,
    rule_id: RuleId,
    worker_boxes: list[Box],
    ablation_phrase: str,
    **run_kwargs,
) -> tuple[str, float]:
    """Text-ablation analog of answer_rule for the grounding proxy (Phase 2.1).

    Replaces the rule's specific object phrase (e.g. "hard hat") with a concept-
    free placeholder (e.g. "object") and re-decides the rule, reusing the
    baseline worker detection because the image is unchanged. This is the
    closest a grounding proxy can come to a native-VQA "remove the query token"
    ablation; it is NOT equivalent -- it measures whether the specific object
    word (rather than any detectable object) drives the compliance answer -- and
    its results are always reported separately from native-VQA text ablation.

    Returns (answer, graded_score).
    """
    ablated_object_boxes, _ = _detect(model, processor, image, ablation_phrase, **run_kwargs)
    max_dim = max(image.width, image.height)
    if rule_id == "rule_4":
        threshold = PROXIMITY_FRACTION * max_dim
        return _rule4_verdict(worker_boxes, ablated_object_boxes, distance_threshold=threshold)
    threshold = PRESENCE_PROXIMITY_FRACTION[rule_id] * max_dim
    return _presence_verdict(worker_boxes, ablated_object_boxes, distance_threshold=threshold)


def answer_rule(
    model,
    processor,
    image: Image.Image,
    rule_id: RuleId,
    phrasing_index: int = 0,
    **run_kwargs,
) -> AnswerResult:
    """Answer one safety rule for one image. See module docstring for the method."""
    if rule_id == "rule_4":
        return _answer_rule_4(model, processor, image, **run_kwargs)
    return _answer_presence_rule(model, processor, image, rule_id, phrasing_index, **run_kwargs)


def _answer_presence_rule(
    model,
    processor,
    image: Image.Image,
    rule_id: RuleId,
    phrasing_index: int = 0,
    **run_kwargs,
) -> AnswerResult:
    """Evaluate PPE or edge protection relative to each detected worker.

    Two independent grounding calls are necessary because Florence-2 does not
    expose a relational VQA task. The result is compliant only when every
    detected worker has at least one sufficiently close safety-object box.
    """
    phrase = RULE_QUERIES[rule_id][phrasing_index]
    t0 = time.perf_counter()
    worker_boxes, worker_conf = _detect(model, processor, image, PERSON_PHRASE, **run_kwargs)
    object_boxes, object_conf = _detect(model, processor, image, phrase, **run_kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    # Both detections are required for the proxy decision, so their simple mean
    # is used as the run-level confidence signal. It remains uncalibrated.
    confidence = (worker_conf + object_conf) / 2

    if not worker_boxes:
        # No worker detected at all -- can't ask "does every worker have X",
        # so fall back to the scene-level existence check as a degenerate case.
        answer, graded = _presence_verdict([], object_boxes, distance_threshold=0.0)
        return AnswerResult(
            answer=answer,
            boxes=object_boxes,
            confidence=confidence,
            inference_ms=elapsed_ms,
            worker_boxes=[],
            object_boxes=object_boxes,
            graded_score=graded,
        )

    # Scale thresholds by the longer image dimension so the same rule behaves
    # consistently across source images with different pixel resolutions.
    distance_threshold = PRESENCE_PROXIMITY_FRACTION[rule_id] * max(image.width, image.height)
    answer, graded = _presence_verdict(worker_boxes, object_boxes, distance_threshold=distance_threshold)
    return AnswerResult(
        answer=answer,
        boxes=worker_boxes + object_boxes,
        confidence=confidence,
        inference_ms=elapsed_ms,
        worker_boxes=worker_boxes,
        object_boxes=object_boxes,
        graded_score=graded,
    )


def _answer_rule_4(model, processor, image: Image.Image, **run_kwargs) -> AnswerResult:
    """Flag struck-by risk when any worker is near any excavator.

    The nested pairwise comparison intentionally implements an existential
    safety condition: one hazardous worker-excavator pairing is enough to mark
    the entire scene as a hazard.
    """
    worker_phrase, excavator_phrase = RULE_4_PROXIMITY_PAIR
    t0 = time.perf_counter()
    worker_boxes, worker_conf = _detect(model, processor, image, worker_phrase, **run_kwargs)
    excavator_boxes, excavator_conf = _detect(model, processor, image, excavator_phrase, **run_kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    distance_threshold = PROXIMITY_FRACTION * max(image.width, image.height)
    answer, graded = _rule4_verdict(worker_boxes, excavator_boxes, distance_threshold=distance_threshold)
    confidence = (worker_conf + excavator_conf) / 2
    return AnswerResult(
        answer=answer,
        boxes=worker_boxes + excavator_boxes,
        confidence=confidence,
        inference_ms=elapsed_ms,
        worker_boxes=worker_boxes,
        object_boxes=excavator_boxes,
        graded_score=graded,
    )
