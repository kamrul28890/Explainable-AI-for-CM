"""Robustness: do answers and explanations survive perturbation?

Day 5 found that masking a sample's only detected worker reroutes
_answer_presence_rule into its scene-level fallback branch, flipping the
answer for reasons disconnected from the safety concept being tested. The
same risk applies here: a perturbation that happens to make the worker
undetectable can flip the answer the same way. `worker_lost` flags that
directly so it isn't silently folded into a generic "answer changed" rate.
"""

from dataclasses import dataclass

from PIL import Image

from xai_pilot.inference import AnswerResult, answer_rule
from xai_pilot.prompts import RuleId
from xai_pilot.regions import iou


@dataclass
class RobustnessResult:
    answer_changed: bool
    confidence_drop: float
    object_box_iou: float  # nan if either side has no object box to compare
    worker_lost: bool  # baseline detected >=1 worker, perturbed run detected none


def _robustness_from_results(baseline: AnswerResult, perturbed: AnswerResult) -> RobustnessResult:
    object_box_iou = (
        iou(baseline.object_boxes[0], perturbed.object_boxes[0])
        if baseline.object_boxes and perturbed.object_boxes
        else float("nan")
    )
    return RobustnessResult(
        answer_changed=perturbed.answer != baseline.answer,
        confidence_drop=baseline.confidence - perturbed.confidence,
        object_box_iou=object_box_iou,
        worker_lost=bool(baseline.worker_boxes) and not perturbed.worker_boxes,
    )


def evaluate(
    model,
    processor,
    perturbed_image: Image.Image,
    rule_id: RuleId,
    baseline: AnswerResult,
    **run_kwargs,
) -> RobustnessResult:
    """Rerun answer_rule on perturbed_image and compare to the baseline result."""
    perturbed = answer_rule(model, processor, perturbed_image, rule_id, **run_kwargs)
    return _robustness_from_results(baseline, perturbed)
