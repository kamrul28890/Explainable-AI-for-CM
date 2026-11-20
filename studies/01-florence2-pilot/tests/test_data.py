"""Unit tests for dataset classification and experimental rule assignment."""

from xai_pilot.data import (
    assign_compliant_rule,
    assign_rule_id,
    classify_image,
    image_classes,
    scene_has_excavator,
    select_balanced_sample,
)

BASE_ROW = {
    "image_id": "test",
    "rule_1_violation": None,
    "rule_2_violation": None,
    "rule_3_violation": None,
    "rule_4_violation": None,
}


def _row(**violations):
    row = dict(BASE_ROW)
    row.update(violations)
    return row


def test_compliant_when_no_violations():
    primary_class, violated = classify_image(_row())
    assert primary_class == "compliant"
    assert violated == []


def test_rule_1_is_ppe_violation():
    row = _row(rule_1_violation={"bounding_box": [[0.1, 0.1, 0.2, 0.2]], "reason": "no hard hat"})
    primary_class, violated = classify_image(row)
    assert primary_class == "ppe_violation"
    assert violated == ["rule_1_violation"]


def test_rule_2_is_fall_hazard():
    row = _row(rule_2_violation={"bounding_box": [[0.1, 0.1, 0.2, 0.2]], "reason": "no harness"})
    primary_class, violated = classify_image(row)
    assert primary_class == "fall_hazard"
    assert violated == ["rule_2_violation"]


def test_rule_3_is_fall_hazard():
    row = _row(rule_3_violation={"bounding_box": [[0.1, 0.1, 0.2, 0.2]], "reason": "no guardrail"})
    primary_class, violated = classify_image(row)
    assert primary_class == "fall_hazard"
    assert violated == ["rule_3_violation"]


def test_rule_4_is_struck_by_risk():
    row = _row(rule_4_violation={"bounding_box": [[0.1, 0.1, 0.2, 0.2]], "reason": "in blind spot"})
    primary_class, violated = classify_image(row)
    assert primary_class == "struck_by_risk"
    assert violated == ["rule_4_violation"]


def test_rule_1_takes_priority_over_rule_4():
    row = _row(
        rule_1_violation={"bounding_box": [[0.1, 0.1, 0.2, 0.2]], "reason": "no vest"},
        rule_4_violation={"bounding_box": [[0.3, 0.3, 0.4, 0.4]], "reason": "in blind spot"},
    )
    primary_class, violated = classify_image(row)
    assert primary_class == "ppe_violation"
    assert set(violated) == {"rule_1_violation", "rule_4_violation"}


def test_assign_rule_id_for_violation_classes():
    assert assign_rule_id("ppe_violation", ["rule_1_violation"]) == "rule_1"
    assert assign_rule_id("struck_by_risk", ["rule_4_violation"]) == "rule_4"


def test_assign_rule_id_picks_first_matching_rule_for_fall_hazard():
    assert assign_rule_id("fall_hazard", ["rule_2_violation", "rule_3_violation"]) == "rule_2"
    assert assign_rule_id("fall_hazard", ["rule_3_violation"]) == "rule_3"


def test_assign_rule_id_ignores_unrelated_violated_rules():
    # rule_1 + rule_4 both violated, but primary_class is ppe_violation (priority order)
    assert assign_rule_id("ppe_violation", ["rule_1_violation", "rule_4_violation"]) == "rule_1"


def test_assign_rule_id_round_robins_compliant_samples():
    assigned = [assign_rule_id("compliant", [], i) for i in range(8)]
    assert assigned == ["rule_1", "rule_2", "rule_3", "rule_4"] * 2


# --- Phase 1.2: multi-label classification ------------------------------------


def test_image_classes_compliant_when_no_violations():
    assert image_classes(_row()) == ["compliant"]


def test_image_classes_single_violation():
    row = _row(rule_4_violation={"reason": "blind spot"})
    assert image_classes(row) == ["struck_by_risk"]


def test_image_classes_returns_full_set_not_just_priority():
    # The exact case priority-collapse discarded: a struck_by_risk image that
    # also has a higher-priority PPE violation. Multi-label keeps both.
    row = _row(
        rule_1_violation={"reason": "no hat"},
        rule_4_violation={"reason": "blind spot"},
    )
    assert image_classes(row) == ["ppe_violation", "struck_by_risk"]


def test_image_classes_dedups_fall_hazard_from_two_rules():
    # rule_2 and rule_3 both map to fall_hazard; it appears once.
    row = _row(
        rule_2_violation={"reason": "no harness"},
        rule_3_violation={"reason": "no guardrail"},
    )
    assert image_classes(row) == ["fall_hazard"]


# --- Phase 1.2: multi-label-aware sampling ------------------------------------


def _fake_ds(rows):
    """A streamable dataset stand-in: select_balanced_sample only iterates it."""
    return list(rows)


def test_priority_sampling_assigns_single_rule_per_image():
    rows = [
        _row(image_id=f"c{i}") for i in range(4)  # compliant
    ] + [
        _row(image_id="v1", rule_1_violation={"reason": "x"}),
    ]
    samples = select_balanced_sample(_fake_ds(rows), n_per_class=1, seed=0, labeling="priority")
    for s in samples:
        # Frozen behavior: exactly one assigned rule, mirrored into the list field.
        assert s.assigned_rule_ids == [s.assigned_rule_id]


def test_multilabel_sampling_tests_every_violated_rule():
    row = _row(
        image_id="v1",
        rule_1_violation={"reason": "no hat"},
        rule_4_violation={"reason": "blind spot"},
    )
    samples = select_balanced_sample(_fake_ds([row]), n_per_class=1, seed=0, labeling="multilabel")
    by_id = {s.image_id: s for s in samples}
    assert "v1" in by_id
    assert set(by_id["v1"].assigned_rule_ids) == {"rule_1", "rule_4"}


def test_multilabel_recovers_struck_by_risk_hidden_by_priority():
    # One image violates rule_1 (ppe, higher priority) AND rule_4 (struck_by).
    # Under priority its primary_class is ppe_violation, so struck_by_risk would
    # have zero samples. Multi-label must surface it under struck_by_risk too.
    rows = [
        _row(image_id="hidden", rule_1_violation={"reason": "x"}, rule_4_violation={"reason": "y"}),
    ]
    priority = select_balanced_sample(_fake_ds(rows), n_per_class=5, seed=0, labeling="priority")
    multilabel = select_balanced_sample(_fake_ds(rows), n_per_class=5, seed=0, labeling="multilabel")

    struck_priority = [s for s in priority if "struck_by_risk" in image_classes_of(s)]
    # priority: the single image is classified ppe_violation only
    assert all(s.primary_class == "ppe_violation" for s in priority)
    # multilabel: the same image is now testable for rule_4
    assert any("rule_4" in s.assigned_rule_ids for s in multilabel)


def image_classes_of(sample):
    """Helper: recover the class set a sample carries (test-only convenience)."""
    return {sample.primary_class}


def test_multilabel_does_not_duplicate_an_image_across_classes():
    # An image in two class reservoirs must appear once, with rules unioned.
    row = _row(image_id="dup", rule_1_violation={"reason": "x"}, rule_4_violation={"reason": "y"})
    samples = select_balanced_sample(_fake_ds([row]), n_per_class=5, seed=0, labeling="multilabel")
    ids = [s.image_id for s in samples]
    assert ids.count("dup") == 1


# --- Phase 1.3: compliant-sample context matching -----------------------------


def test_scene_has_excavator_true_when_boxes_present():
    assert scene_has_excavator({"excavator": [[0.1, 0.1, 0.4, 0.4]]}) is True


def test_scene_has_excavator_false_when_empty_or_missing():
    assert scene_has_excavator({"excavator": []}) is False
    assert scene_has_excavator({}) is False


def test_assign_compliant_round_robin_cycles_all_four_rules():
    assigned = [assign_compliant_rule(False, i, mode="round_robin") for i in range(8)]
    rules = [r for r, _ in assigned]
    absent = [a for _, a in assigned]
    assert rules == ["rule_1", "rule_2", "rule_3", "rule_4"] * 2
    assert absent == [False] * 8  # round-robin does not assess context


def test_context_matched_routes_excavator_scene_to_rule_4():
    rule, context_absent = assign_compliant_rule(True, 0, mode="context_matched")
    assert rule == "rule_4"
    assert context_absent is False  # confirmed struck-by context


def test_context_matched_never_assigns_vacuous_rule_4_without_excavator():
    assigned = [assign_compliant_rule(False, i, mode="context_matched") for i in range(9)]
    rules = [r for r, _ in assigned]
    absent = [a for _, a in assigned]
    assert "rule_4" not in rules  # no vacuous struck-by test
    assert set(rules) <= {"rule_1", "rule_2", "rule_3"}
    assert all(absent)  # metadata cannot confirm PPE/height/edge context


def test_context_matched_sampling_end_to_end():
    rows = [
        _row(image_id="exc", excavator=[[0.1, 0.1, 0.4, 0.4]]),   # compliant, has excavator
        _row(image_id="flat", excavator=[]),                       # compliant, no excavator
    ]
    samples = select_balanced_sample(
        _fake_ds(rows), n_per_class=5, seed=0,
        labeling="priority", compliant_assignment="context_matched",
    )
    by_id = {s.image_id: s for s in samples}
    assert by_id["exc"].assigned_rule_id == "rule_4"
    assert by_id["exc"].context_absent is False
    assert by_id["flat"].assigned_rule_id != "rule_4"
    assert by_id["flat"].context_absent is True


def test_context_absent_defaults_false_in_round_robin():
    rows = [_row(image_id="c1")]
    samples = select_balanced_sample(_fake_ds(rows), n_per_class=5, seed=0)
    assert samples[0].context_absent is False
