"""Descriptive Accuracy: does masking the top explanation region flip the answer?

For each sample, reruns answer_rule on the image with its top-1 (then
top-1+top-2) standardized region masked out, and checks whether the rule's
answer changes from the unmasked baseline. A region that is genuinely
driving the decision should flip the answer when removed; if masking never
changes anything, the region wasn't load-bearing for the proxy's answer.
"""

from dataclasses import dataclass

from PIL import Image

from xai_pilot.inference import AnswerResult, answer_rule
from xai_pilot.prompts import RuleId
from xai_pilot.regions import Region, mask_region


@dataclass
class DescriptiveAccuracyResult:
    """Answer and confidence changes for cumulative top-region masks.

    Phase 1.4 adds the worker-loss fields: masking a region can make the worker
    undetectable on the rerun, which reroutes _answer_presence_rule into its
    scene-level fallback and flips the answer for a reason unrelated to the
    safety concept. `worker_lost_*` records that the rerun lost a worker the
    baseline had; `flip_due_to_worker_loss_*` is the subset of answer flips that
    co-occur with worker loss, so a "genuine" (de-contaminated) flip rate can be
    reported alongside the raw one.
    """
    answer_changed_top1: bool
    answer_changed_top2: bool
    confidence_drop_top1: float
    confidence_drop_top2: float
    masked_answer_top1: str
    masked_answer_top2: str
    worker_lost_top1: bool = False
    worker_lost_top2: bool = False
    flip_due_to_worker_loss_top1: bool = False
    flip_due_to_worker_loss_top2: bool = False


def _descriptive_from_results(
    baseline: AnswerResult, result_top1: AnswerResult, result_top2: AnswerResult
) -> DescriptiveAccuracyResult:
    """Compute all descriptive-accuracy signals from three normalized results.

    Pure function (no model call), so the worker-loss logic is unit-testable in
    isolation. A worker is "lost" when the baseline detected at least one worker
    but the masked rerun detected none; a flip is attributed to worker loss only
    when the answer changed AND the worker was lost.
    """
    changed_top1 = result_top1.answer != baseline.answer
    changed_top2 = result_top2.answer != baseline.answer
    worker_lost_top1 = bool(baseline.worker_boxes) and not result_top1.worker_boxes
    worker_lost_top2 = bool(baseline.worker_boxes) and not result_top2.worker_boxes
    return DescriptiveAccuracyResult(
        answer_changed_top1=changed_top1,
        answer_changed_top2=changed_top2,
        confidence_drop_top1=baseline.confidence - result_top1.confidence,
        confidence_drop_top2=baseline.confidence - result_top2.confidence,
        masked_answer_top1=result_top1.answer,
        masked_answer_top2=result_top2.answer,
        worker_lost_top1=worker_lost_top1,
        worker_lost_top2=worker_lost_top2,
        flip_due_to_worker_loss_top1=changed_top1 and worker_lost_top1,
        flip_due_to_worker_loss_top2=changed_top2 and worker_lost_top2,
    )


def evaluate(
    model,
    processor,
    image: Image.Image,
    rule_id: RuleId,
    baseline: AnswerResult,
    top_regions: list[Region],
    **run_kwargs,
) -> DescriptiveAccuracyResult:
    """Mask top-1, then top-1+top-2 regions; rerun answer_rule after each."""
    # The top-2 condition is cumulative by design: it removes both the first
    # and second ranked regions rather than testing region two in isolation.
    top1_masked = mask_region(image, top_regions[0].box, mode="black")
    result_top1 = answer_rule(model, processor, top1_masked, rule_id, **run_kwargs)

    if len(top_regions) > 1:
        top2_masked = mask_region(top1_masked, top_regions[1].box, mode="black")
        result_top2 = answer_rule(model, processor, top2_masked, rule_id, **run_kwargs)
    else:
        # Only one region existed at all -- the top-2 condition degenerates to top-1.
        result_top2 = result_top1

    return _descriptive_from_results(baseline, result_top1, result_top2)
