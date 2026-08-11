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
from xai_pilot.regions import iou, normalized_centroid_distance


@dataclass
class RobustnessResult:
    """Comparison signals between one baseline and one perturbed inference."""
    answer_changed: bool
    confidence_drop: float
    object_box_iou: float  # nan if either side has no object box to compare
    worker_lost: bool  # baseline detected >=1 worker, perturbed run detected none
    # Phase 1.4: the answer flip is attributable to worker loss (answer changed
    # AND the worker became undetectable), so a "genuine" robustness rate can be
    # reported by excluding these from the raw answer-change rate.
    flip_due_to_worker_loss: bool = False
    # Phase 2.4: size-invariant explanation drift (normalized centroid distance
    # of the object box, NaN if either side lacks a box or image_size is unknown)
    # plus an explicit disappearance flag, so a vanished box registers as maximal
    # drift instead of dropping silently out of the IoU denominator.
    object_centroid_drift: float = float("nan")
    object_disappeared: bool = False


def _robustness_from_results(
    baseline: AnswerResult, perturbed: AnswerResult, image_size: tuple[int, int] | None = None
) -> RobustnessResult:
    """Compare normalized results without performing another model call.

    The first safety-object box is used because the pilot proxy and reports
    consistently treat it as the representative object detection. Missing
    boxes produce NaN rather than being interpreted as zero overlap. When
    `image_size` is given, a size-invariant centroid drift is also computed.
    """
    both_objects = bool(baseline.object_boxes) and bool(perturbed.object_boxes)
    object_box_iou = (
        iou(baseline.object_boxes[0], perturbed.object_boxes[0]) if both_objects else float("nan")
    )
    object_centroid_drift = (
        normalized_centroid_distance(baseline.object_boxes[0], perturbed.object_boxes[0], image_size)
        if both_objects and image_size is not None
        else float("nan")
    )
    answer_changed = perturbed.answer != baseline.answer
    worker_lost = bool(baseline.worker_boxes) and not perturbed.worker_boxes
    return RobustnessResult(
        answer_changed=answer_changed,
        confidence_drop=baseline.confidence - perturbed.confidence,
        object_box_iou=object_box_iou,
        worker_lost=worker_lost,
        flip_due_to_worker_loss=answer_changed and worker_lost,
        object_centroid_drift=object_centroid_drift,
        object_disappeared=bool(baseline.object_boxes) and not perturbed.object_boxes,
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
    return _robustness_from_results(baseline, perturbed, image_size=perturbed_image.size)
