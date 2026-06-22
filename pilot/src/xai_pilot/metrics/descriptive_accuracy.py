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
    answer_changed_top1: bool
    answer_changed_top2: bool
    confidence_drop_top1: float
    confidence_drop_top2: float
    masked_answer_top1: str
    masked_answer_top2: str


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
    top1_masked = mask_region(image, top_regions[0].box, mode="black")
    result_top1 = answer_rule(model, processor, top1_masked, rule_id, **run_kwargs)

    if len(top_regions) > 1:
        top2_masked = mask_region(top1_masked, top_regions[1].box, mode="black")
        result_top2 = answer_rule(model, processor, top2_masked, rule_id, **run_kwargs)
    else:
        # Only one region existed at all -- the top-2 condition degenerates to top-1.
        result_top2 = result_top1

    return DescriptiveAccuracyResult(
        answer_changed_top1=result_top1.answer != baseline.answer,
        answer_changed_top2=result_top2.answer != baseline.answer,
        confidence_drop_top1=baseline.confidence - result_top1.confidence,
        confidence_drop_top2=baseline.confidence - result_top2.confidence,
        masked_answer_top1=result_top1.answer,
        masked_answer_top2=result_top2.answer,
    )
