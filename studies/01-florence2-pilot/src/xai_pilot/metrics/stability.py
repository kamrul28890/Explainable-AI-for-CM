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
from xai_pilot.regions import Box, Region, iou, normalized_centroid_distance


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


def object_presence_rate(results: list[AnswerResult]) -> float:
    """Fraction of reruns in which the safety object was detected at all
    (Scale-up Phase 2.3).

    The pilot's overlap scores returned NaN and silently dropped a sample when
    the object box appeared in some reruns and vanished in others -- the worst
    kind of instability. This turns that into a number: 1.0 means the object was
    present in every rerun, 0.0 in none, and an intermediate value flags a
    flickering detection. NaN only for an empty rerun list.
    """
    if not results:
        return float("nan")
    return sum(1 for r in results if r.object_boxes) / len(results)


def mean_pairwise_centroid_distance(boxes: list[Box], image_size: tuple[int, int]) -> float:
    """Mean pairwise box-centroid distance across reruns, normalized by the
    image diagonal (Scale-up Phase 2.3).

    A size-invariant companion to IoU: IoU collapses quickly for small boxes
    that jitter by a few pixels, penalizing small PPE detections purely for
    their size, whereas normalized centroid distance stays small when the
    detections sit in the same place regardless of box size. 0.0 means the
    centroids coincide across all reruns (stable); larger means more drift. NaN
    for fewer than two boxes.
    """
    if len(boxes) < 2:
        return float("nan")
    dists = [
        normalized_centroid_distance(a, b, image_size)
        for a, b in itertools.combinations(boxes, 2)
    ]
    return sum(dists) / len(dists)


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
