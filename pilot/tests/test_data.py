from xai_pilot.data import assign_rule_id, classify_image

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
