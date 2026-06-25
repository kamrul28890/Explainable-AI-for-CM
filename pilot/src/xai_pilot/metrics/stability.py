"""Stability: do reruns of the same sample agree with each other?

Florence-2's default decoding (beam search, no sampling) is deterministic --
three literal reruns of the pipeline would be bit-identical and "100%
stable" by construction, which would prove nothing. The actual stability
test reruns with `do_sample=True, num_beams=1, temperature>0` instead (see
inference.answer_rule's **run_kwargs), so the n reruns can genuinely
disagree. The deterministic beam-search answer (already logged in Days
3-5's CSVs) is reported alongside, separately, as a labeled "decoding
ceiling" reference -- not mixed into the stability score itself.
"""

import itertools

from PIL import Image

from xai_pilot.inference import AnswerResult, answer_rule
from xai_pilot.prompts import RuleId
from xai_pilot.regions import Region, iou


def run_n_times(
    model,
    processor,
    image: Image.Image,
    rule_id: RuleId,
    n: int = 3,
    temperature: float = 0.7,
    **extra_kwargs,
) -> list[AnswerResult]:
    """Rerun answer_rule n times with sampling-based decoding (do_sample=True, num_beams=1)."""
    # Keep complete AnswerResult objects because stability is evaluated both at
    # the categorical answer level and at the grounding-region level.
    return [
        answer_rule(model, processor, image, rule_id, do_sample=True, num_beams=1, temperature=temperature, **extra_kwargs)
        for _ in range(n)
    ]


def answer_agreement_rate(results: list[AnswerResult]) -> float:
    """Fraction of all rerun pairs that produced the same final answer."""
    answers = [r.answer for r in results]
    pairs = list(itertools.combinations(answers, 2))
    if not pairs:
        return float("nan")
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def region_overlap_score(top_regions: list[Region]) -> float:
    """Mean pairwise IoU of the top-1 explanation region's box across reruns.

    nan if any rerun fell back to the grid (no model box at all that run) --
    an arbitrary unranked grid cell isn't a meaningful thing to IoU-compare,
    same reasoning as Day 6's grid-fallback exclusion.
    """
    if any(r.source == "grid" for r in top_regions):
        return float("nan")
    pairs = list(itertools.combinations([r.box for r in top_regions], 2))
    if not pairs:
        return float("nan")
    return sum(iou(a, b) for a, b in pairs) / len(pairs)
