"""Unit tests for the pure proxy-verdict helpers, incl. graded score (Phase 2.1)."""

import math

from xai_pilot.inference import _presence_verdict, _rule4_verdict, graded_score_for


# --- presence rules (rule_1/2/3): graded = fraction of workers covered --------


def test_presence_all_workers_covered_is_compliant_score_one():
    workers = [(0, 0, 10, 10), (50, 50, 60, 60)]
    objects = [(2, 2, 8, 8), (52, 52, 58, 58)]
    answer, score = _presence_verdict(workers, objects, distance_threshold=0.0)
    assert answer == "compliant"
    assert score == 1.0


def test_presence_partial_coverage_is_violation_with_fractional_score():
    workers = [(0, 0, 10, 10), (50, 50, 60, 60)]
    objects = [(2, 2, 8, 8)]  # only first worker covered
    answer, score = _presence_verdict(workers, objects, distance_threshold=0.0)
    assert answer == "violation"
    assert score == 0.5


def test_presence_no_objects_is_violation_score_zero():
    answer, score = _presence_verdict([(0, 0, 10, 10)], [], distance_threshold=0.0)
    assert answer == "violation"
    assert score == 0.0


def test_presence_no_workers_falls_back_to_scene_level_score_nan():
    # Object present -> scene-level fallback answers compliant; graded undefined.
    answer, score = _presence_verdict([], [(0, 0, 10, 10)], distance_threshold=0.0)
    assert answer == "compliant"
    assert math.isnan(score)
    answer2, score2 = _presence_verdict([], [], distance_threshold=0.0)
    assert answer2 == "violation"
    assert math.isnan(score2)


# --- rule_4: graded = safe fraction (1 - fraction of workers near excavator) ---


def test_rule4_worker_near_excavator_is_hazard():
    workers = [(0, 0, 10, 10)]
    excavators = [(5, 5, 15, 15)]
    answer, score = _rule4_verdict(workers, excavators, distance_threshold=0.0)
    assert answer == "hazard"
    assert score == 0.0  # 1 worker, near -> safe fraction 0


def test_rule4_no_worker_near_is_safe_score_one():
    workers = [(0, 0, 10, 10)]
    excavators = [(100, 100, 110, 110)]
    answer, score = _rule4_verdict(workers, excavators, distance_threshold=1.0)
    assert answer == "safe"
    assert score == 1.0


def test_rule4_no_workers_is_safe_score_nan():
    answer, score = _rule4_verdict([], [(0, 0, 10, 10)], distance_threshold=0.0)
    assert answer == "safe"
    assert math.isnan(score)


# --- graded_score_for: recompute a baseline graded score from saved boxes -----


def test_graded_score_for_presence_rule_uses_rule_threshold():
    # Worker and hard hat overlap -> covered -> score 1.0 for a presence rule.
    workers = [(100, 100, 200, 300)]
    objects = [(120, 100, 160, 130)]
    score = graded_score_for("rule_1", workers, objects, image_size=(1000, 1000))
    assert score == 1.0


def test_graded_score_for_rule4_is_safe_fraction():
    # One worker, far from the excavator -> safe fraction 1.0.
    workers = [(0, 0, 10, 10)]
    excavators = [(900, 900, 950, 950)]
    score = graded_score_for("rule_4", workers, excavators, image_size=(1000, 1000))
    assert score == 1.0
