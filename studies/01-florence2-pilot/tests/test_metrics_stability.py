"""Unit tests for answer agreement and explanation-region overlap."""

import math

from xai_pilot.inference import AnswerResult
from xai_pilot.metrics.stability import (
    answer_agreement_rate,
    mean_pairwise_centroid_distance,
    object_presence_rate,
    region_overlap_score,
)
from xai_pilot.regions import Region


def test_answer_agreement_rate_all_same_is_one():
    results = [AnswerResult(answer="compliant") for _ in range(3)]
    assert answer_agreement_rate(results) == 1.0


def test_answer_agreement_rate_all_different_is_zero():
    results = [
        AnswerResult(answer="compliant"),
        AnswerResult(answer="violation"),
        AnswerResult(answer="violation"),
    ]
    # pairs: (compliant,violation)=disagree, (compliant,violation)=disagree, (violation,violation)=agree
    assert abs(answer_agreement_rate(results) - 1 / 3) < 1e-9


def test_answer_agreement_rate_single_result_is_nan():
    import math

    assert math.isnan(answer_agreement_rate([AnswerResult(answer="compliant")]))


def test_region_overlap_score_identical_boxes_is_one():
    regions = [Region(box=(0, 0, 10, 10), source="model") for _ in range(3)]
    assert region_overlap_score(regions) == 1.0


def test_region_overlap_score_disjoint_boxes_is_zero():
    regions = [
        Region(box=(0, 0, 10, 10), source="model"),
        Region(box=(100, 100, 110, 110), source="model"),
    ]
    assert region_overlap_score(regions) == 0.0


def test_region_overlap_score_nan_when_any_run_used_grid_fallback():
    import math

    regions = [
        Region(box=(0, 0, 10, 10), source="model"),
        Region(box=(0, 0, 25, 20), source="grid", label="grid"),
    ]
    assert math.isnan(region_overlap_score(regions))


# --- Phase 2.3: object-presence stability -------------------------------------
# The pilot returned NaN (silently dropping the sample) when the object box
# appeared in some reruns and vanished in others -- the worst instability. This
# counts it as a number instead.


def test_object_presence_rate_all_present_is_one():
    results = [AnswerResult(answer="x", object_boxes=[(0, 0, 5, 5)]) for _ in range(4)]
    assert object_presence_rate(results) == 1.0


def test_object_presence_rate_all_absent_is_zero():
    results = [AnswerResult(answer="x", object_boxes=[]) for _ in range(4)]
    assert object_presence_rate(results) == 0.0


def test_object_presence_rate_flickering_is_fractional():
    results = [
        AnswerResult(answer="x", object_boxes=[(0, 0, 5, 5)]),
        AnswerResult(answer="x", object_boxes=[]),
        AnswerResult(answer="x", object_boxes=[(0, 0, 5, 5)]),
        AnswerResult(answer="x", object_boxes=[]),
    ]
    assert object_presence_rate(results) == 0.5


def test_object_presence_rate_empty_is_nan():
    assert math.isnan(object_presence_rate([]))


# --- Phase 2.3: size-invariant centroid-distance overlap ----------------------
# IoU collapses fast for small boxes that jitter slightly; normalized centroid
# distance stays small, so small PPE boxes aren't penalized purely by size.


def test_centroid_distance_identical_boxes_is_zero():
    boxes = [(0, 0, 10, 10), (0, 0, 10, 10)]
    assert mean_pairwise_centroid_distance(boxes, image_size=(100, 100)) == 0.0


def test_centroid_distance_small_jitter_stays_small_where_iou_would_collapse():
    # Two tiny 4px boxes shifted by 2px: IoU is low, but the normalized centroid
    # distance is tiny (2px / ~141px diagonal).
    boxes = [(0, 0, 4, 4), (2, 0, 6, 4)]
    d = mean_pairwise_centroid_distance(boxes, image_size=(100, 100))
    assert d < 0.05


def test_centroid_distance_single_box_is_nan():
    assert math.isnan(mean_pairwise_centroid_distance([(0, 0, 10, 10)], image_size=(100, 100)))
