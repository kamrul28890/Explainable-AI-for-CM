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
from xai_pilot.regions import all_boxes_covered, boxes_overlap_or_close

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
    answer: str
    boxes: list[Box] = field(default_factory=list)
    confidence: float = float("nan")
    inference_ms: float = 0.0
    worker_boxes: list[Box] = field(default_factory=list)
    object_boxes: list[Box] = field(default_factory=list)


def _detect(model, processor, image: Image.Image, phrase: str, **run_kwargs):
    _, parsed, confidence = run_task(
        model, processor, image, DETECTION_TASK, text_input=phrase, **run_kwargs
    )
    det = parsed[DETECTION_TASK]
    boxes = [tuple(b) for b in det["bboxes"]]
    return boxes, confidence


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
    phrase = RULE_QUERIES[rule_id][phrasing_index]
    t0 = time.perf_counter()
    worker_boxes, worker_conf = _detect(model, processor, image, PERSON_PHRASE, **run_kwargs)
    object_boxes, object_conf = _detect(model, processor, image, phrase, **run_kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    confidence = (worker_conf + object_conf) / 2

    if not worker_boxes:
        # No worker detected at all -- can't ask "does every worker have X",
        # so fall back to the scene-level existence check as a degenerate case.
        answer = "compliant" if object_boxes else "violation"
        return AnswerResult(
            answer=answer,
            boxes=object_boxes,
            confidence=confidence,
            inference_ms=elapsed_ms,
            worker_boxes=[],
            object_boxes=object_boxes,
        )

    distance_threshold = PRESENCE_PROXIMITY_FRACTION[rule_id] * max(image.width, image.height)
    covered = all_boxes_covered(worker_boxes, object_boxes, distance_threshold=distance_threshold)
    answer = "compliant" if covered else "violation"
    return AnswerResult(
        answer=answer,
        boxes=worker_boxes + object_boxes,
        confidence=confidence,
        inference_ms=elapsed_ms,
        worker_boxes=worker_boxes,
        object_boxes=object_boxes,
    )


def _answer_rule_4(model, processor, image: Image.Image, **run_kwargs) -> AnswerResult:
    worker_phrase, excavator_phrase = RULE_4_PROXIMITY_PAIR
    t0 = time.perf_counter()
    worker_boxes, worker_conf = _detect(model, processor, image, worker_phrase, **run_kwargs)
    excavator_boxes, excavator_conf = _detect(model, processor, image, excavator_phrase, **run_kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    distance_threshold = PROXIMITY_FRACTION * max(image.width, image.height)
    hazard = any(
        boxes_overlap_or_close(w, e, distance_threshold=distance_threshold)
        for w in worker_boxes
        for e in excavator_boxes
    )
    answer = "hazard" if hazard else "safe"
    confidence = (worker_conf + excavator_conf) / 2
    return AnswerResult(
        answer=answer,
        boxes=worker_boxes + excavator_boxes,
        confidence=confidence,
        inference_ms=elapsed_ms,
        worker_boxes=worker_boxes,
        object_boxes=excavator_boxes,
    )
