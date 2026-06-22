"""VQA-via-grounding: answer a safety rule for one image using Florence-2.

Florence-2-base-ft has no free-form VQA task token, so each rule is answered
by grounding a phrase instead of asking a yes/no question (see
pilot-plan.md, Design Decision #1). Rules 1-3 are presence checks via
<OPEN_VOCABULARY_DETECTION>; rule 4 is a proximity check between two
independently grounded objects (worker, excavator).
"""

import time
from dataclasses import dataclass, field

from PIL import Image

from xai_pilot.model import run_task
from xai_pilot.prompts import RULE_4_PROXIMITY_PAIR, RULE_QUERIES, RuleId
from xai_pilot.regions import boxes_overlap_or_close

Box = tuple[float, float, float, float]

DETECTION_TASK = "<OPEN_VOCABULARY_DETECTION>"
PROXIMITY_FRACTION = 0.05  # blind-spot distance threshold as a fraction of max(image dim)


@dataclass
class AnswerResult:
    answer: str
    boxes: list[Box] = field(default_factory=list)
    confidence: float = float("nan")
    inference_ms: float = 0.0


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

    phrase = RULE_QUERIES[rule_id][phrasing_index]
    t0 = time.perf_counter()
    boxes, confidence = _detect(model, processor, image, phrase, **run_kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    answer = "compliant" if boxes else "violation"
    return AnswerResult(answer=answer, boxes=boxes, confidence=confidence, inference_ms=elapsed_ms)


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
    )
