"""Unit tests for the three bounded-completeness verdicts."""

import pytest

from xai_pilot.metrics.completeness import (
    classify_sample,
    classify_sample_worker_loss_corrected,
    multi_hazard_verdict,
)


def test_grid_fallback_is_no_usable_explanation_even_if_answer_changed():
    assert classify_sample("grid", answer_changed_top1=True, answer_changed_top2=True) == "no_usable_explanation"


def test_model_region_with_no_answer_change_is_weak():
    assert classify_sample("model", answer_changed_top1=False, answer_changed_top2=False) == "explanation_weak"


def test_model_region_with_top1_change_is_supported():
    assert classify_sample("model", answer_changed_top1=True, answer_changed_top2=False) == "explanation_supported"


def test_model_region_with_only_top2_change_is_supported():
    assert classify_sample("model", answer_changed_top1=False, answer_changed_top2=True) == "explanation_supported"


# --- Phase 1.4: worker-loss-corrected verdict ---------------------------------


def test_corrected_equals_raw_when_no_worker_loss():
    v = classify_sample_worker_loss_corrected(
        "model", True, False, flip_due_to_worker_loss_top1=False, flip_due_to_worker_loss_top2=False
    )
    assert v == "explanation_supported"


def test_corrected_downgrades_supported_driven_only_by_worker_loss():
    # The only flip (top-1) is spurious worker-loss; corrected verdict is weak.
    v = classify_sample_worker_loss_corrected(
        "model", True, False, flip_due_to_worker_loss_top1=True, flip_due_to_worker_loss_top2=False
    )
    assert v == "explanation_weak"


def test_corrected_keeps_supported_when_a_genuine_flip_remains():
    # top-1 flip is worker-loss, but top-2 flip is genuine -> still supported.
    v = classify_sample_worker_loss_corrected(
        "model", True, True, flip_due_to_worker_loss_top1=True, flip_due_to_worker_loss_top2=False
    )
    assert v == "explanation_supported"


def test_corrected_grid_is_still_no_usable():
    v = classify_sample_worker_loss_corrected(
        "grid", True, True, flip_due_to_worker_loss_top1=True, flip_due_to_worker_loss_top2=True
    )
    assert v == "no_usable_explanation"


# --- Phase 2.5: multi-hazard completeness -------------------------------------


def test_multi_hazard_complete_when_every_hazard_supported():
    assert multi_hazard_verdict({"rule_1": True, "rule_4": True}) == "complete"


def test_multi_hazard_partial_when_some_supported():
    assert multi_hazard_verdict({"rule_1": True, "rule_4": False}) == "partial"


def test_multi_hazard_none_when_no_hazard_supported():
    assert multi_hazard_verdict({"rule_1": False, "rule_3": False}) == "none"


def test_multi_hazard_requires_at_least_two_hazards():
    with pytest.raises(ValueError):
        multi_hazard_verdict({"rule_1": True})
