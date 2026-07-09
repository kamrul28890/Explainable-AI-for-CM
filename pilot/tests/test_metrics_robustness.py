"""Unit tests for baseline-to-perturbation robustness comparisons."""

import math

from xai_pilot.inference import AnswerResult
from xai_pilot.metrics.robustness import _robustness_from_results


def test_answer_changed_true_when_answers_differ():
    baseline = AnswerResult(answer="violation")
    perturbed = AnswerResult(answer="compliant")
    result = _robustness_from_results(baseline, perturbed)
    assert result.answer_changed is True


def test_answer_changed_false_when_answers_match():
    baseline = AnswerResult(answer="violation")
    perturbed = AnswerResult(answer="violation")
    result = _robustness_from_results(baseline, perturbed)
    assert result.answer_changed is False


def test_confidence_drop_is_baseline_minus_perturbed():
    baseline = AnswerResult(answer="violation", confidence=0.9)
    perturbed = AnswerResult(answer="violation", confidence=0.6)
    result = _robustness_from_results(baseline, perturbed)
    assert abs(result.confidence_drop - 0.3) < 1e-9


def test_object_box_iou_nan_when_perturbed_has_no_object_box():
    baseline = AnswerResult(answer="violation", object_boxes=[(0, 0, 10, 10)])
    perturbed = AnswerResult(answer="violation", object_boxes=[])
    result = _robustness_from_results(baseline, perturbed)
    assert math.isnan(result.object_box_iou)


def test_object_box_iou_computed_when_both_sides_have_a_box():
    baseline = AnswerResult(answer="violation", object_boxes=[(0, 0, 10, 10)])
    perturbed = AnswerResult(answer="violation", object_boxes=[(0, 0, 10, 10)])
    result = _robustness_from_results(baseline, perturbed)
    assert result.object_box_iou == 1.0


def test_worker_lost_true_when_baseline_had_worker_and_perturbed_does_not():
    baseline = AnswerResult(answer="violation", worker_boxes=[(0, 0, 5, 5)])
    perturbed = AnswerResult(answer="compliant", worker_boxes=[])
    result = _robustness_from_results(baseline, perturbed)
    assert result.worker_lost is True


def test_worker_lost_false_when_baseline_never_had_a_worker():
    baseline = AnswerResult(answer="violation", worker_boxes=[])
    perturbed = AnswerResult(answer="violation", worker_boxes=[])
    result = _robustness_from_results(baseline, perturbed)
    assert result.worker_lost is False


# --- Phase 1.4: flip attributable to worker loss ------------------------------


def test_flip_due_to_worker_loss_true_when_flip_coincides_with_worker_loss():
    baseline = AnswerResult(answer="violation", worker_boxes=[(0, 0, 5, 5)])
    perturbed = AnswerResult(answer="compliant", worker_boxes=[])
    result = _robustness_from_results(baseline, perturbed)
    assert result.flip_due_to_worker_loss is True


def test_flip_due_to_worker_loss_false_when_flip_without_worker_loss():
    # A genuine flip: the worker is still detected, the answer changed anyway.
    baseline = AnswerResult(answer="violation", worker_boxes=[(0, 0, 5, 5)])
    perturbed = AnswerResult(answer="compliant", worker_boxes=[(0, 0, 5, 5)])
    result = _robustness_from_results(baseline, perturbed)
    assert result.flip_due_to_worker_loss is False


def test_flip_due_to_worker_loss_false_when_worker_lost_but_no_flip():
    baseline = AnswerResult(answer="violation", worker_boxes=[(0, 0, 5, 5)])
    perturbed = AnswerResult(answer="violation", worker_boxes=[])
    result = _robustness_from_results(baseline, perturbed)
    assert result.flip_due_to_worker_loss is False
