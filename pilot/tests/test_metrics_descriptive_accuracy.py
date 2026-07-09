"""Unit tests for the descriptive-accuracy pure core, incl. worker-loss (Phase 1.4)."""

from xai_pilot.inference import AnswerResult
from xai_pilot.metrics.descriptive_accuracy import _descriptive_from_results


def _res(answer, worker_boxes=None, confidence=0.5):
    return AnswerResult(answer=answer, worker_boxes=worker_boxes or [], confidence=confidence)


def test_answer_changed_flags_track_baseline_difference():
    baseline = _res("violation", worker_boxes=[(0, 0, 5, 5)])
    top1 = _res("compliant", worker_boxes=[(0, 0, 5, 5)])
    top2 = _res("violation", worker_boxes=[(0, 0, 5, 5)])
    r = _descriptive_from_results(baseline, top1, top2)
    assert r.answer_changed_top1 is True
    assert r.answer_changed_top2 is False


def test_worker_loss_flip_is_flagged_when_top1_mask_loses_worker():
    baseline = _res("violation", worker_boxes=[(0, 0, 5, 5)])
    top1 = _res("compliant", worker_boxes=[])  # worker undetectable after mask
    r = _descriptive_from_results(baseline, top1, top1)
    assert r.worker_lost_top1 is True
    assert r.flip_due_to_worker_loss_top1 is True


def test_genuine_flip_not_attributed_to_worker_loss():
    # Answer flipped but the worker is still detected -> a genuine flip.
    baseline = _res("violation", worker_boxes=[(0, 0, 5, 5)])
    top1 = _res("compliant", worker_boxes=[(0, 0, 5, 5)])
    r = _descriptive_from_results(baseline, top1, top1)
    assert r.worker_lost_top1 is False
    assert r.flip_due_to_worker_loss_top1 is False


def test_worker_loss_without_flip_is_not_a_worker_loss_flip():
    baseline = _res("violation", worker_boxes=[(0, 0, 5, 5)])
    top1 = _res("violation", worker_boxes=[])  # worker lost, but answer unchanged
    r = _descriptive_from_results(baseline, top1, top1)
    assert r.worker_lost_top1 is True
    assert r.flip_due_to_worker_loss_top1 is False


def test_no_worker_loss_when_baseline_had_no_worker():
    baseline = _res("violation", worker_boxes=[])
    top1 = _res("compliant", worker_boxes=[])
    r = _descriptive_from_results(baseline, top1, top1)
    assert r.worker_lost_top1 is False
    assert r.flip_due_to_worker_loss_top1 is False
